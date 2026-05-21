"""Manual check: does VOSK live up to its "real-time, runs on CPU" claim,
and how does its transcription accuracy compare to the current production
STT (gpt-4o-transcribe, app/services/transcription.py)?

Ground truth = a fixed set of known English sentences, spoken aloud via our
own REAL RealtimeApiSpeaker (proven verbatim in scripts/check_speaker.py),
so we have exact-text audio without needing a labeled dataset. Each sample
is then transcribed by:
  1. VOSK (one or more local offline models), timed to compute a Real-Time
     Factor (RTF = decode_time / audio_duration; RTF < 1.0 means faster
     than real time -- i.e. the "real-time on CPU" claim holds).
  2. The current gpt-4o-transcribe API, for a same-audio accuracy baseline.

WER/CER (via jiwer) are computed against the known ground-truth text for
both. This is an evaluation script only -- it does not change which STT
the app uses.

Usage: python -m scripts.check_vosk_accuracy
Requires: OPENAI_API_KEY in .env (for ground-truth TTS + the comparison
baseline), and VOSK models unzipped under tmp/vosk_models/.
"""

import asyncio
import io
import json
import re
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import jiwer
import numpy as np
from scipy.signal import resample_poly
from vosk import KaldiRecognizer, Model, SetLogLevel

from app.services.realtime_tts import SAMPLE_RATE as TTS_SAMPLE_RATE
from app.services.realtime_tts import RealtimeApiSpeaker
from app.services.transcription import transcribe_audio

SetLogLevel(-1)  # silence Kaldi's own logging so our output stays readable

VOSK_SAMPLE_RATE = 16000
MODELS_DIR = Path(__file__).resolve().parent.parent / "tmp" / "vosk_models"
VOSK_MODELS = {
    "vosk-small-en-us-0.15 (40MB)": MODELS_DIR / "vosk-model-small-en-us-0.15",
}

# Domain-relevant sentences (sales-coaching register), mixing plain prose,
# numbers, and slightly longer structure to stress the models a bit rather
# than testing on trivially easy audio.
SAMPLES = [
    "Welcome back, let's continue practicing your objection handling skills today.",
    "If the customer says the price is too high, don't panic, acknowledge and reframe the value first.",
    "In twenty twenty six, more than five thousand sales reps completed this program successfully.",
    "Confidence, clarity, and consistency are the three pillars of a strong sales pitch.",
    "Can you walk me through how you would close this deal by Friday?",
]


@dataclass
class SampleAudio:
    text: str
    pcm16_24k: bytes  # raw PCM16 mono @ 24kHz, straight from the TTS


@dataclass
class TranscriptionResult:
    hypothesis: str
    seconds: float  # wall-clock decode/API time


def norm(text: str) -> str:
    """Lowercase + strip punctuation before scoring. VOSK never emits
    punctuation/capitalization while gpt-4o-transcribe does, so comparing
    raw strings would penalize VOSK for formatting, not word accuracy."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def to_wav_bytes(pcm16: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)
    return buf.getvalue()


def resample_24k_to_16k(pcm16_24k: bytes) -> bytes:
    samples = np.frombuffer(pcm16_24k, dtype=np.int16)
    resampled = resample_poly(samples, up=2, down=3)  # 24000 * 2/3 = 16000
    return resampled.astype(np.int16).tobytes()


async def generate_ground_truth_audio(sentences: list[str]) -> list[SampleAudio]:
    samples: list[SampleAudio] = []
    async with RealtimeApiSpeaker() as speaker:
        for text in sentences:
            chunks = [chunk async for chunk in speaker.speak(text)]
            samples.append(SampleAudio(text=text, pcm16_24k=b"".join(chunks)))
            print(f"  generated audio: {text!r}")
    return samples


def transcribe_with_vosk(model: Model, pcm16_16k: bytes) -> TranscriptionResult:
    rec = KaldiRecognizer(model, VOSK_SAMPLE_RATE)
    rec.SetWords(False)

    start = time.perf_counter()
    chunk_size = 4000  # bytes, matches common vosk examples (~125ms @ 16kHz/16-bit)
    for i in range(0, len(pcm16_16k), chunk_size):
        rec.AcceptWaveform(pcm16_16k[i : i + chunk_size])
    final = json.loads(rec.FinalResult())
    elapsed = time.perf_counter() - start

    return TranscriptionResult(hypothesis=final.get("text", ""), seconds=elapsed)


async def transcribe_with_gpt4o(pcm16_24k: bytes) -> TranscriptionResult:
    wav_bytes = to_wav_bytes(pcm16_24k, TTS_SAMPLE_RATE)
    start = time.perf_counter()
    text = await transcribe_audio(wav_bytes, filename="sample.wav")
    elapsed = time.perf_counter() - start
    return TranscriptionResult(hypothesis=text, seconds=elapsed)


def audio_duration_seconds(pcm16: bytes, sample_rate: int) -> float:
    return len(pcm16) / 2 / sample_rate


def print_report(rows: list[dict]) -> None:
    header = f"{'model':<32} {'sample':<8} {'RTF':>6} {'WER':>7} {'CER':>7}"
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        rtf = f"{row['rtf']:.3f}" if row["rtf"] is not None else "n/a"
        print(f"{row['model']:<32} {row['sample']:<8} {rtf:>6} {row['wer']:>7.3f} {row['cer']:>7.3f}")


async def main() -> None:
    missing = [name for name, path in VOSK_MODELS.items() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing VOSK model dirs, run the download step first: {missing}")

    print("Generating ground-truth audio via RealtimeApiSpeaker (verbatim TTS)...")
    samples = await generate_ground_truth_audio(SAMPLES)

    rows: list[dict] = []
    all_refs: list[str] = []
    all_hyps_by_model: dict[str, list[str]] = {name: [] for name in VOSK_MODELS}
    all_hyps_gpt4o: list[str] = []

    print("\nLoading VOSK models...")
    loaded_models = {name: Model(str(path)) for name, path in VOSK_MODELS.items()}

    for idx, sample in enumerate(samples, start=1):
        sample_label = f"#{idx}"
        all_refs.append(sample.text)
        duration = audio_duration_seconds(sample.pcm16_24k, TTS_SAMPLE_RATE)
        pcm16k = resample_24k_to_16k(sample.pcm16_24k)

        for model_name, model in loaded_models.items():
            result = transcribe_with_vosk(model, pcm16k)
            all_hyps_by_model[model_name].append(result.hypothesis)
            print(f"    [{model_name}] -> {result.hypothesis!r}")
            rows.append(
                {
                    "model": model_name,
                    "sample": sample_label,
                    "rtf": result.seconds / duration,
                    "wer": jiwer.wer(norm(sample.text), norm(result.hypothesis)),
                    "cer": jiwer.cer(norm(sample.text), norm(result.hypothesis)),
                }
            )

        print(f"  transcribing sample {sample_label} with gpt-4o-transcribe...")
        gpt4o_result = await transcribe_with_gpt4o(sample.pcm16_24k)
        all_hyps_gpt4o.append(gpt4o_result.hypothesis)
        print(f"    [gpt-4o-transcribe] -> {gpt4o_result.hypothesis!r}")
        rows.append(
            {
                "model": "gpt-4o-transcribe (API)",
                "sample": sample_label,
                "rtf": gpt4o_result.seconds / duration,
                "wer": jiwer.wer(norm(sample.text), norm(gpt4o_result.hypothesis)),
                "cer": jiwer.cer(norm(sample.text), norm(gpt4o_result.hypothesis)),
            }
        )

    print_report(rows)

    norm_refs = [norm(r) for r in all_refs]
    print("\nOverall (all samples pooled, lowercased + punctuation-stripped):")
    for model_name in VOSK_MODELS:
        hyps = [norm(h) for h in all_hyps_by_model[model_name]]
        wer = jiwer.wer(norm_refs, hyps)
        cer = jiwer.cer(norm_refs, hyps)
        print(f"  {model_name:<32} WER={wer:.3f}  CER={cer:.3f}")
    hyps = [norm(h) for h in all_hyps_gpt4o]
    wer = jiwer.wer(norm_refs, hyps)
    cer = jiwer.cer(norm_refs, hyps)
    print(f"  {'gpt-4o-transcribe (API)':<32} WER={wer:.3f}  CER={cer:.3f}")

    print(
        "\nNote: RTF = decode_time / audio_duration. RTF < 1.0 means the model "
        "processes audio faster than it plays -- i.e. real-time-capable on this CPU. "
        "VOSK's RTF here excludes model load time (loaded once above, as a real "
        "deployment would); gpt-4o-transcribe's 'RTF' is dominated by network/API "
        "latency, not decode speed, so it is not a fair CPU-speed comparison -- it's "
        "shown only for context."
    )


if __name__ == "__main__":
    asyncio.run(main())
