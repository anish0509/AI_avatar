"""Phase 2 experiment harness for Bug 4 (trailing phantom-token).

Replays the REAL captured mic WAVs in tmp/ (ground truth = the filename)
through the REAL RealtimeApiTranscriber under each candidate configuration,
ONE parameter changed at a time, and scores every run so configs can be
compared against the baseline. See bug-report-streaming-stt.md.

Why this design (the three things synthetic tests got wrong before):
  * Real audio only -- the bug never reproduced on synthetic clips, so we
    replay the user's actual recordings, nothing generated.
  * Real-time pacing -- audio is fed in ~100ms chunks with wall-clock sleeps
    so server_vad segments the clip the way it did live (fast-dumping the WAV
    changes VAD behaviour, and VAD segmentation is what causes this bug).
  * Repeated runs per clip -- the model is non-deterministic on short clips,
    so each clip is run --runs times per config and results are aggregated
    into rates, not read off a single pass.

Metrics per config (all derived from the transcript event stream, so nothing
in the production interface has to change):
  * hallucination rate      -- runs where an EXTRA trailing token appears
                               after the correct word(s) ("hello" -> "hello bam")
  * accuracy                 -- runs where the ground-truth word(s) came back
                               correctly (a phantom token can co-occur with a
                               correct core, exactly the bug)
  * missing first words      -- runs where the first ground-truth word is gone
                               (guards against a raised threshold clipping onset)
  * perceived latency        -- mean seconds from first audio to first delta
  * false VAD activations    -- runs that produced >1 finalized segment for one
                               short utterance (segment-count proxy)
  * missed speech detections -- runs that produced 0 segments / empty transcript

Usage (hits the REAL OpenAI API -- needs OPENAI_API_KEY in .env):
  python -m scripts.stt_experiment --list
  python -m scripts.stt_experiment --experiments baseline --runs 3
  python -m scripts.stt_experiment --experiments baseline,noise_far --runs 3
  python -m scripts.stt_experiment                      # full matrix, --runs 3
  python -m scripts.stt_experiment --clips "hello*,go*,hi*,react*"

Start with `--experiments baseline` and CONFIRM the phantom token reproduces
before trusting any comparison -- if baseline shows a ~0 hallucination rate on
your captures, the dataset can't measure a fix and needs more/noisier clips.
"""

import argparse
import asyncio
import csv
import re
import string
import wave
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import TextIO

from app.core.config import settings
from app.services.realtime_stt import INPUT_SAMPLE_RATE, RealtimeApiTranscriber

TMP_DIR = Path(__file__).resolve().parent.parent / "tmp"
CHUNK_BYTES = 4800  # ~100ms @ 24kHz/16-bit mono -- matches the browser's send size
# Faster-than-real-time feed: server_vad measures silence in AUDIO time (the
# gaps are in the samples themselves), not wall-clock arrival, so feeding 5x
# faster preserves segmentation while cutting replay time on long clips. (0.05
# i.e. 2x was already proven fine in check_realtime_stt.py.)
CHUNK_INTERVAL_SECONDS = 0.02
TRAILING_SILENCE_SECONDS = 1.2  # let server_vad finalize (as check_realtime_stt.py does)
POST_FEED_WAIT_SECONDS = 1.0  # after audio, give VAD a moment to close the segment
DRAIN_GRACE_SECONDS = 3.0  # once fed, wait this long for a (possibly phantom 2nd) segment
FEED_EVENT_TIMEOUT_SECONDS = 15  # while still feeding, cap the wait for the first event

EXCLUDED_STEMS = {"check_speaker", "connectivity_check"}

# The five knobs the harness overrides; anything not named in a config is reset
# to its "unset" default so exactly ONE parameter differs from baseline.
OVERRIDE_DEFAULTS: dict[str, object] = {
    "realtime_noise_reduction": "",
    "realtime_vad_threshold": None,
    "realtime_prefix_padding_ms": None,
    "realtime_silence_duration_ms": None,
    "realtime_transcribe_prompt": "",
}

CONFIGS: dict[str, dict[str, object]] = {
    "baseline": {},
    "noise_far": {"realtime_noise_reduction": "far_field"},
    "noise_near": {"realtime_noise_reduction": "near_field"},
    "thr_055": {"realtime_vad_threshold": 0.55},
    "thr_060": {"realtime_vad_threshold": 0.60},
    "thr_065": {"realtime_vad_threshold": 0.65},
    "sil_700": {"realtime_silence_duration_ms": 700},
    "sil_500": {"realtime_silence_duration_ms": 500},
    "sil_300": {"realtime_silence_duration_ms": 300},
    "sil_200": {"realtime_silence_duration_ms": 200},
    "prefix_300": {"realtime_prefix_padding_ms": 300},
    "prompt": {"realtime_transcribe_prompt": "Transcribe short English words exactly as spoken."},
    # Combination (only after the singles): the winning noise_reduction plus a
    # MILD threshold bump (0.60, not the accuracy-hurting 0.65).
    "noise_far_thr060": {"realtime_noise_reduction": "far_field", "realtime_vad_threshold": 0.60},
}


def normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into word tokens for comparison."""
    cleaned = text.lower().translate(str.maketrans("", "", string.punctuation))
    return cleaned.split()


def ground_truth_from_name(path: Path) -> str:
    """The spoken word(s) are the filename; strip a trailing '(2)' repeat marker."""
    stem = re.sub(r"\(\d+\)$", "", path.stem).strip()
    return stem


@dataclass
class RunResult:
    transcript: str
    segments: int  # number of finalized transcription events
    first_delta_latency: float | None


def _contiguous_index(got: list[str], gt: list[str]) -> int:
    """Index where gt appears as a contiguous run inside got, else -1."""
    if not gt:
        return -1
    for i in range(len(got) - len(gt) + 1):
        if got[i : i + len(gt)] == gt:
            return i
    return -1


@dataclass
class ClipScore:
    # The bug: correct word(s) came back PLUS extra phantom token(s).
    phantom_extra: bool
    phantom_trailing: bool  # subset of phantom_extra that is specifically trailing
    core_correct: bool  # the expected word(s) transcribed correctly (extras allowed)
    core_wrong: bool  # word mis-transcribed (acoustic error, not a knob target)
    false_activation: bool
    missed_speech: bool
    latency: float | None


def score_run(ground_truth: str, run: RunResult) -> ClipScore:
    gt = normalize(ground_truth)
    got = normalize(run.transcript)

    missed = run.segments == 0 or len(got) == 0
    idx = _contiguous_index(got, gt)
    core_correct = (not missed) and idx != -1
    # Phantom extra token(s): the word is there correctly, but with junk added
    # before and/or after it (lead: "Anyway Hello"; trail: "You Sure").
    phantom_extra = core_correct and len(got) > len(gt)
    phantom_trailing = phantom_extra and idx == 0  # gt at the front -> extra is trailing
    # Core wrong: got something, but the expected word isn't in it (e.g. "go" ->
    # "Golf", "i" -> "Ay"). This is acoustic mis-recognition of an ultra-short,
    # context-free clip -- a DIFFERENT problem the VAD/noise knobs won't fix.
    core_wrong = (not missed) and idx == -1

    return ClipScore(
        phantom_extra=phantom_extra,
        phantom_trailing=phantom_trailing,
        core_correct=core_correct,
        core_wrong=core_wrong,
        false_activation=run.segments > 1,
        missed_speech=missed,
        latency=run.first_delta_latency,
    )


@dataclass
class ConfigReport:
    name: str
    total_runs: int = 0
    phantom_extra: int = 0
    phantom_trailing: int = 0
    core_correct: int = 0
    core_wrong: int = 0
    false_activations: int = 0
    missed: int = 0
    latencies: list[float] = field(default_factory=list)
    # Per-clip phantom examples, for eyeballing what actually changed.
    examples: list[tuple[str, str]] = field(default_factory=list)

    def add(self, ground_truth: str, run: RunResult, score: ClipScore) -> None:
        self.total_runs += 1
        self.phantom_extra += score.phantom_extra
        self.phantom_trailing += score.phantom_trailing
        self.core_correct += score.core_correct
        self.core_wrong += score.core_wrong
        self.false_activations += score.false_activation
        self.missed += score.missed_speech
        if score.latency is not None:
            self.latencies.append(score.latency)
        if score.phantom_extra:
            self.examples.append((ground_truth, run.transcript))

    def _rate(self, n: int) -> str:
        return f"{100 * n / self.total_runs:4.0f}%" if self.total_runs else "  - "

    @property
    def mean_latency(self) -> str:
        return f"{sum(self.latencies) / len(self.latencies):.2f}s" if self.latencies else "  -  "

    def row(self) -> str:
        return (
            f"{self.name:<12} {self._rate(self.phantom_extra)}  {self._rate(self.phantom_trailing)}  "
            f"{self._rate(self.core_correct)}  {self._rate(self.core_wrong)}  "
            f"{self.mean_latency:>6}  {self._rate(self.false_activations)}  {self._rate(self.missed)}"
        )


def load_wav_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != INPUT_SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(
                f"{path.name}: expected mono/16-bit/{INPUT_SAMPLE_RATE}Hz, "
                f"got {w.getnchannels()}ch/{8 * w.getsampwidth()}bit/{w.getframerate()}Hz"
            )
        return w.readframes(w.getnframes())


async def run_clip(pcm: bytes) -> RunResult:
    """Stream one clip through the real transcriber, real-time paced, and
    collect the final transcript, segment count, and first-delta latency."""
    pcm = pcm + b"\x00\x00" * int(INPUT_SAMPLE_RATE * TRAILING_SILENCE_SECONDS)

    async with RealtimeApiTranscriber() as transcriber:
        loop = asyncio.get_event_loop()
        start = loop.time()
        feed_done = asyncio.Event()

        async def feed() -> None:
            for i in range(0, len(pcm), CHUNK_BYTES):
                await transcriber.send_audio(pcm[i : i + CHUNK_BYTES])
                await asyncio.sleep(CHUNK_INTERVAL_SECONDS)
            await asyncio.sleep(POST_FEED_WAIT_SECONDS)  # let VAD notice end of speech
            feed_done.set()

        feeder = asyncio.create_task(feed())
        events = transcriber.events().__aiter__()
        transcript_parts: list[str] = []
        segments = 0
        first_delta_latency: float | None = None

        # Read events as they arrive. While still feeding, allow a long wait for
        # the first event; once the audio is in, drain any remaining (possibly
        # phantom second) segments within a short grace window, then stop -- so
        # a clip takes ~feed_time + grace, not the full per-clip timeout.
        try:
            while True:
                timeout = DRAIN_GRACE_SECONDS if feed_done.is_set() else FEED_EVENT_TIMEOUT_SECONDS
                try:
                    event = await asyncio.wait_for(events.__anext__(), timeout)
                except (asyncio.TimeoutError, StopAsyncIteration):
                    break
                if event.kind == "delta" and first_delta_latency is None:
                    first_delta_latency = loop.time() - start
                elif event.kind == "final":
                    transcript_parts.append(event.text)
                    segments += 1
        finally:
            feeder.cancel()

    return RunResult(
        transcript=" ".join(p for p in transcript_parts if p.strip()),
        segments=segments,
        first_delta_latency=first_delta_latency,
    )


def apply_config(overrides: dict[str, object]) -> None:
    for field_name, default in OVERRIDE_DEFAULTS.items():
        setattr(settings, field_name, overrides.get(field_name, default))


def select_clips(clip_globs: list[str] | None) -> list[Path]:
    clips = [
        p
        for p in sorted(TMP_DIR.glob("*.wav"))
        if p.stem not in EXCLUDED_STEMS
    ]
    if clip_globs:
        clips = [p for p in clips if any(fnmatch(p.name, g.strip()) for g in clip_globs)]
    return clips


def _flag(score: ClipScore) -> str:
    if score.phantom_trailing:
        return "PHANTOM-TRAIL"
    if score.phantom_extra:
        return "PHANTOM-LEAD"
    if score.missed_speech:
        return "MISS"
    if score.core_wrong:
        return "core-wrong"
    return "ok"


async def run_experiments(
    config_names: list[str],
    clips: list[Path],
    runs: int,
    concurrency: int,
    csv_writer: "csv.writer | None" = None,
    csv_file: TextIO | None = None,
) -> list[ConfigReport]:
    saved = {name: getattr(settings, name) for name in OVERRIDE_DEFAULTS}
    loaded = [(clip, ground_truth_from_name(clip), load_wav_pcm(clip)) for clip in clips]
    reports: list[ConfigReport] = []

    async def one_run(clip: Path, pcm: bytes, sem: asyncio.Semaphore) -> tuple[Path, RunResult | Exception]:
        async with sem:  # cap concurrent Realtime connections
            last_exc: Exception | None = None
            for attempt in range(3):  # tolerate transient network/DNS blips
                try:
                    return clip, await run_clip(pcm)
                except Exception as exc:  # one bad run must not kill the matrix
                    last_exc = exc
                    await asyncio.sleep(2 * (attempt + 1))
            return clip, last_exc  # type: ignore[return-value]

    try:
        for name in config_names:
            # apply_config mutates GLOBAL settings, so configs must run
            # sequentially; all clips WITHIN a config share it and run concurrently.
            apply_config(CONFIGS[name])
            report = ConfigReport(name=name)
            print(f"\n=== {name}  ({CONFIGS[name] or 'defaults'}) ===")
            sem = asyncio.Semaphore(concurrency)
            gt_by_clip = {clip: gt for clip, gt, _ in loaded}
            tasks = [
                asyncio.create_task(one_run(clip, pcm, sem))
                for clip, _, pcm in loaded
                for _ in range(runs)
            ]
            for clip, outcome in await asyncio.gather(*tasks):
                gt = gt_by_clip[clip]
                if isinstance(outcome, Exception):
                    print(f"  {clip.name} run: ERROR {outcome}")
                    continue
                score = score_run(gt, outcome)
                report.add(gt, outcome, score)
                flag = _flag(score)
                print(f"  {clip.name:<32} gt={gt!r:<28} -> {outcome.transcript!r:<34} [{flag}]")
                if csv_writer is not None and csv_file is not None:
                    # Flush per row so results survive truncation/buffering -- the
                    # stdout capture kept losing the tail on long runs.
                    csv_writer.writerow(
                        [name, clip.name, gt, outcome.transcript, flag, outcome.segments,
                         f"{outcome.first_delta_latency:.3f}" if outcome.first_delta_latency else ""]
                    )
                    csv_file.flush()
            reports.append(report)
    finally:
        for name, value in saved.items():
            setattr(settings, name, value)
    return reports


def print_table(reports: list[ConfigReport]) -> None:
    print("\n" + "=" * 88)
    print("COMPARISON (rates over all clip-runs). THE BUG = 'extra'/'trail'; lower is better.")
    print("coreOK = word transcribed correctly (extras allowed); coreBad = acoustic mis-read.")
    print("=" * 88)
    print(
        f"{'config':<12} {'extra':>5}  {'trail':>5}  {'coreOK':>6}  {'coreBad':>7}  "
        f"{'latncy':>6}  {'falseVAD':>8}  {'missed':>6}"
    )
    print("-" * 88)
    for r in reports:
        print(r.row())
    print("-" * 88)
    if reports:
        base = reports[0]
        print(f"\n(baseline = {base.name}: {base.phantom_extra}/{base.total_runs} runs had a phantom extra token)")
        if base.phantom_extra == 0:
            print(
                "WARNING: baseline shows NO trailing hallucinations on this dataset -- "
                "there is nothing to measure a fix against. Capture more / noisier short "
                "clips before trusting any comparison below."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bug 4 STT config experiment harness")
    parser.add_argument("--experiments", help="comma-separated config names (default: all)")
    parser.add_argument("--clips", help="comma-separated filename globs to include (default: all)")
    parser.add_argument("--runs", type=int, default=3, help="runs per clip per config (default 3)")
    parser.add_argument("--concurrency", type=int, default=5, help="concurrent Realtime connections (default 5)")
    parser.add_argument("--csv", help="append per-run results to this CSV (flushed per row)")
    parser.add_argument("--list", action="store_true", help="list configs and clips, then exit")
    args = parser.parse_args()

    clip_globs = args.clips.split(",") if args.clips else None
    clips = select_clips(clip_globs)

    if args.experiments:
        config_names = [n.strip() for n in args.experiments.split(",")]
        unknown = [n for n in config_names if n not in CONFIGS]
        if unknown:
            parser.error(f"unknown experiment(s): {unknown}. Known: {list(CONFIGS)}")
    else:
        config_names = list(CONFIGS)
    # Always keep baseline first so the table's reference row is the baseline.
    if "baseline" in config_names:
        config_names = ["baseline"] + [n for n in config_names if n != "baseline"]

    if args.list:
        print("Configs:", ", ".join(CONFIGS))
        print(f"\nClips ({len(clips)}):")
        for c in clips:
            print(f"  {c.name:<34} gt={ground_truth_from_name(c)!r}")
        return

    if not clips:
        parser.error(f"no matching .wav clips in {TMP_DIR}")

    print(f"Configs: {config_names}")
    print(f"Clips: {len(clips)} | runs/clip: {args.runs} | concurrency: {args.concurrency} | "
          f"total sessions: {len(config_names) * len(clips) * args.runs}")

    csv_file: TextIO | None = None
    csv_writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["config", "clip", "ground_truth", "transcript", "flag", "segments", "first_delta_s"])
        csv_file.flush()
    try:
        reports = asyncio.run(
            run_experiments(config_names, clips, args.runs, args.concurrency, csv_writer, csv_file)
        )
    finally:
        if csv_file is not None:
            csv_file.close()
    print_table(reports)


if __name__ == "__main__":
    main()
