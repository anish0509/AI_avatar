const WS_PATH = "/ws/voice";
const STT_WS_PATH = "/ws/transcribe";
const SAMPLE_RATE = 24000; // server contract: PCM16 mono @ 24kHz (both playback and mic capture)

let audioCtx = null;
let analyser = null;
let analyserData = null;
let nextStartTime = 0;
let scheduledSources = [];

let ws = null;
let turnId = 0;
let turnReachedTerminal = false;

// Each "Ask" opens a brand-new WebSocket (see connect()), so the server has
// no transport-level way to know two turns are the same conversation. This
// client-minted id is sent with every prompt and echoed back in "meta" so
// the agent's per-thread memory (server-side) can actually accumulate
// across turns. Only meaningful when ANSWER_SOURCE=agent; "gpt"/"rag" ignore
// it. Reset on Clear, since clearing the visible transcript should also
// start a fresh conversation server-side, not just wipe the display.
let threadId = null;

let micStream = null;
let isRecording = false;
let sttSocket = null;
let micAudioCtx = null;
let micSourceNode = null;
let micWorkletNode = null;
let micSilentGain = null;
let committedText = "";
let partialText = "";
let micPcmBuffer = new Float32Array(0);
let micSendChunkSamples = 0;

let canvas, statusEl, answerSourceEl, transcriptEl, formEl, inputEl, askBtn, stopBtn, clearBtn, micBtn;

function getWsUrl(path) {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${window.location.host}${path}`;
}

function ensureAudioContext() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    analyserData = new Uint8Array(analyser.fftSize);
    analyser.connect(audioCtx.destination);
    nextStartTime = audioCtx.currentTime;
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }
}

function closeSocket() {
  if (!ws) return;
  ws.onopen = null;
  ws.onmessage = null;
  ws.onerror = null;
  ws.onclose = null;
  if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
    ws.close();
  }
  ws = null;
}

function stopAllScheduledAudio() {
  for (const source of scheduledSources) {
    try {
      source.stop();
    } catch (_err) {
      // already ended -- fine
    }
  }
  scheduledSources = [];
  if (audioCtx) {
    nextStartTime = audioCtx.currentTime;
  }
}

function setStatus(text, kind) {
  statusEl.textContent = text;
  statusEl.className = `status status--${kind}`;
}

function setBusy(isBusy) {
  askBtn.disabled = isBusy;
  inputEl.disabled = isBusy;
  micBtn.disabled = isBusy;
  stopBtn.hidden = !isBusy;
}

function appendSentence(text) {
  transcriptEl.textContent += (transcriptEl.textContent ? " " : "") + text;
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

// Starts a new turn as a labeled "You: .../Assistant: " pair appended to the
// running transcript, rather than wiping it -- without this, the agent's
// multi-turn memory would work server-side but be invisible/unverifiable in
// the UI, since a follow-up question would show only its own answer with no
// visible trace of the earlier turns it's actually building on.
function appendTurnSeparator(prompt) {
  const prefix = transcriptEl.textContent ? "\n\n" : "";
  transcriptEl.textContent += `${prefix}You: ${prompt}\nAssistant: `;
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

// Renders the per-turn "meta" event so it's always visible, in the browser,
// exactly which answer source produced this reply and (when retrieval ran)
// what it decided. This is the answer to "are we generating from RAG/the
// agent, or directly from the LLM?" -- for all three answer sources.
function showAnswerSource(msg) {
  if (msg.thread_id) {
    threadId = msg.thread_id; // server echoes back the thread it used/minted
  }

  if (msg.answer_source === "agent") {
    if (msg.is_conversational) {
      answerSourceEl.textContent = "Source: AGENT · conversational (answered from history, no retrieval)";
      answerSourceEl.className = "answer-source answer-source--rag";
    } else {
      const verdict = msg.grounded ? "grounded ✓" : "NOT grounded — abstaining (no LLM call)";
      const from = msg.sources && msg.sources.length ? ` · from: ${msg.sources.join(", ")}` : "";
      answerSourceEl.textContent =
        `Source: AGENT · query: "${msg.planner_query}" · top score ${msg.top_score} (floor ${msg.score_floor}) · ${verdict}${from}`;
      answerSourceEl.className = `answer-source answer-source--${msg.grounded ? "rag" : "blocked"}`;
    }
  } else if (msg.answer_source === "rag") {
    const verdict = msg.grounded ? "grounded ✓" : "NOT grounded — abstaining (no LLM call)";
    const from = msg.sources && msg.sources.length ? ` · from: ${msg.sources.join(", ")}` : "";
    answerSourceEl.textContent =
      `Source: RAG · top score ${msg.top_score} (floor ${msg.score_floor}) · ${verdict}${from}`;
    answerSourceEl.className = `answer-source answer-source--${msg.grounded ? "rag" : "blocked"}`;
  } else {
    answerSourceEl.textContent = "Source: GPT — direct LLM, NO retrieval";
    answerSourceEl.className = "answer-source answer-source--gpt";
  }
  answerSourceEl.hidden = false;
}

function playPcmChunk(arrayBuffer) {
  const int16 = new Int16Array(arrayBuffer);
  if (int16.length === 0) return;

  const buffer = audioCtx.createBuffer(1, int16.length, SAMPLE_RATE);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < int16.length; i++) {
    channel[i] = int16[i] / 32768;
  }

  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(analyser);

  const startAt = Math.max(nextStartTime, audioCtx.currentTime);
  source.start(startAt);
  nextStartTime = startAt + buffer.duration;

  scheduledSources.push(source);
}

function handleMessage(event, myTurn) {
  if (myTurn !== turnId) return; // stale message from a superseded turn

  if (typeof event.data === "string") {
    const msg = JSON.parse(event.data);
    if (msg.type === "meta") {
      showAnswerSource(msg);
    } else if (msg.type === "text") {
      appendSentence(msg.text);
    } else if (msg.type === "done") {
      turnReachedTerminal = true;
      setStatus("Done", "idle");
    } else if (msg.type === "error") {
      turnReachedTerminal = true;
      setStatus(`Error: ${msg.detail}`, "error");
    }
  } else {
    setStatus("Speaking…", "speaking");
    playPcmChunk(event.data);
  }
}

function finishTurn(myTurn) {
  if (myTurn !== turnId) return;
  if (!turnReachedTerminal) {
    setStatus("Connection closed unexpectedly", "error");
  }
  setBusy(false);
}

function connect(prompt) {
  turnId += 1;
  const myTurn = turnId;
  turnReachedTerminal = false;

  closeSocket();
  stopAllScheduledAudio();
  appendTurnSeparator(prompt);
  answerSourceEl.hidden = true;
  answerSourceEl.textContent = "";
  setStatus("Connecting…", "idle");
  setBusy(true);

  const socket = new WebSocket(getWsUrl(WS_PATH));
  socket.binaryType = "arraybuffer";
  socket.onopen = () => socket.send(JSON.stringify({ prompt, thread_id: threadId }));
  socket.onmessage = (event) => handleMessage(event, myTurn);
  socket.onerror = () => {
    /* real failures surface via onclose */
  };
  socket.onclose = () => finishTurn(myTurn);

  ws = socket;
}

function handleSubmit(event) {
  event.preventDefault();
  const prompt = inputEl.value.trim();
  if (!prompt) return;
  ensureAudioContext();
  connect(prompt);
}

function handleStop() {
  if (!ws) return;
  turnId += 1; // supersede this turn so any already-queued message is ignored
  turnReachedTerminal = true;
  stopAllScheduledAudio();
  closeSocket();
  setStatus("Stopped", "idle");
  setBusy(false);
}

function handleClear() {
  transcriptEl.textContent = "";
  answerSourceEl.hidden = true;
  answerSourceEl.textContent = "";
  // Starts a genuinely new conversation, not just a visual reset: the next
  // prompt sends no thread_id, so the server mints a fresh one with no
  // memory of anything said before -- otherwise "Clear" would look like a
  // fresh start while the agent silently still remembered the old thread.
  threadId = null;
}

function releaseMic() {
  if (micStream) {
    micStream.getTracks().forEach((track) => track.stop());
    micStream = null;
  }
}

function floatTo16BitPCM(float32Array) {
  const int16Array = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    int16Array[i] = s < 0 ? s * 32768 : s * 32767;
  }
  return int16Array;
}

function resampleTo24k(float32Array, fromRate) {
  // Fast path: AudioContext({sampleRate: 24000}) is honored on most
  // desktop browsers, so this is a no-op in the common case. Falls back to
  // simple linear interpolation on platforms that ignore the hint --
  // skipping this would send audio at the wrong pitch/speed and badly
  // degrade transcription accuracy (the same mismatch that garbled a demo
  // during the VOSK evaluation).
  if (fromRate === SAMPLE_RATE) return float32Array;
  const ratio = SAMPLE_RATE / fromRate;
  const outLength = Math.round(float32Array.length * ratio);
  const result = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const srcIndex = i / ratio;
    const i0 = Math.floor(srcIndex);
    const i1 = Math.min(i0 + 1, float32Array.length - 1);
    const frac = srcIndex - i0;
    result[i] = float32Array[i0] * (1 - frac) + float32Array[i1] * frac;
  }
  return result;
}

async function handleMic() {
  if (isRecording) {
    stopStreamingMic();
    return;
  }
  await startStreamingMic();
}

async function startStreamingMic() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (_err) {
    setStatus("Microphone permission denied", "error");
    return;
  }

  micStream = stream;
  committedText = "";
  partialText = "";
  inputEl.value = "";
  isRecording = true;
  micBtn.classList.add("recording");
  askBtn.disabled = true;
  inputEl.disabled = true;
  setStatus("Connecting…", "idle");

  sttSocket = new WebSocket(getWsUrl(STT_WS_PATH));
  sttSocket.onmessage = handleSttMessage;
  sttSocket.onclose = handleSttSocketClosed;
  sttSocket.onopen = () => beginMicCapture(stream);
}

function sendMicChunk(float32Chunk) {
  const pcm16 = floatTo16BitPCM(resampleTo24k(float32Chunk, micAudioCtx.sampleRate));
  if (sttSocket && sttSocket.readyState === WebSocket.OPEN) {
    sttSocket.send(pcm16.buffer);
  }
}

function flushMicBuffer() {
  if (micPcmBuffer.length > 0) {
    sendMicChunk(micPcmBuffer);
    micPcmBuffer = new Float32Array(0);
  }
}

async function beginMicCapture(stream) {
  micAudioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
  await micAudioCtx.audioWorklet.addModule("/static/pcm-recorder-worklet.js");

  // The worklet posts a message every ~128-sample quantum (~5ms @ 24kHz --
  // ~187/s). Sending each one as its own WebSocket message overwhelmed
  // both the socket and the server->OpenAI relay in real browser testing:
  // a backlog built up and grew without bound, so transcription fell
  // further and further behind and even "stop" got stuck queued behind
  // thousands of unprocessed audio frames. Batching into ~100ms chunks
  // (matching the pacing already proven against the real API in
  // scripts/check_realtime_stt.py) keeps the message rate around 10/s.
  micSendChunkSamples = Math.round(micAudioCtx.sampleRate * 0.1);
  micPcmBuffer = new Float32Array(0);

  micSourceNode = micAudioCtx.createMediaStreamSource(stream);
  micWorkletNode = new AudioWorkletNode(micAudioCtx, "pcm-recorder");
  micWorkletNode.port.onmessage = (event) => {
    const merged = new Float32Array(micPcmBuffer.length + event.data.length);
    merged.set(micPcmBuffer);
    merged.set(event.data, micPcmBuffer.length);
    micPcmBuffer = merged;

    while (micPcmBuffer.length >= micSendChunkSamples) {
      sendMicChunk(micPcmBuffer.slice(0, micSendChunkSamples));
      micPcmBuffer = micPcmBuffer.slice(micSendChunkSamples);
    }
  };

  // AudioWorkletNodes with a live upstream source but no path to the
  // destination can get starved of processing time in some browsers, so
  // route through a zero-gain node instead of leaving the graph dangling
  // -- keeps process() running without playing the mic back out loud.
  micSilentGain = micAudioCtx.createGain();
  micSilentGain.gain.value = 0;
  micSourceNode.connect(micWorkletNode);
  micWorkletNode.connect(micSilentGain);
  micSilentGain.connect(micAudioCtx.destination);

  setStatus("Listening…", "speaking");
}

function stopStreamingMic() {
  isRecording = false;
  micBtn.classList.remove("recording");
  releaseMic();

  if (micWorkletNode) {
    micWorkletNode.port.onmessage = null;
    micWorkletNode.disconnect();
    micWorkletNode = null;
  }
  flushMicBuffer();
  if (micSourceNode) {
    micSourceNode.disconnect();
    micSourceNode = null;
  }
  if (micSilentGain) {
    micSilentGain.disconnect();
    micSilentGain = null;
  }
  if (micAudioCtx) {
    micAudioCtx.close();
    micAudioCtx = null;
  }

  if (sttSocket && sttSocket.readyState === WebSocket.OPEN) {
    sttSocket.send(JSON.stringify({ type: "stop" }));
  }
  setStatus("Transcribing…", "idle");
  // sttSocket itself stays open to receive any trailing delta/final text;
  // the server closes it after sending "done", which fires onclose below.
}

function handleSttMessage(event) {
  const msg = JSON.parse(event.data);
  if (msg.type === "delta") {
    partialText += msg.text;
    inputEl.value = committedText + partialText;
  } else if (msg.type === "final") {
    committedText = committedText ? `${committedText} ${msg.text}` : msg.text;
    partialText = "";
    inputEl.value = committedText;
  } else if (msg.type === "error") {
    setStatus(`Error: ${msg.detail}`, "error");
  }
  // "done" needs no handling here -- the server closes the socket right
  // after sending it, and onclose (handleSttSocketClosed) does cleanup.
}

function handleSttSocketClosed() {
  askBtn.disabled = false;
  inputEl.disabled = false;
  inputEl.focus();
  if (isRecording) {
    // Socket dropped unexpectedly while still actively streaming.
    isRecording = false;
    micBtn.classList.remove("recording");
    releaseMic();
    setStatus("Connection lost", "error");
  } else if (statusEl.className !== "status status--error") {
    setStatus(inputEl.value ? "Review & press Ask" : "Nothing was heard", inputEl.value ? "idle" : "error");
  }
  sttSocket = null;
}

function drawOrb(timestampMs) {
  const ctx2d = canvas.getContext("2d");
  ctx2d.clearRect(0, 0, canvas.width, canvas.height);

  let amplitude = 0;
  if (analyser) {
    analyser.getByteTimeDomainData(analyserData);
    let sumSquares = 0;
    for (const v of analyserData) {
      const d = (v - 128) / 128;
      sumSquares += d * d;
    }
    amplitude = Math.sqrt(sumSquares / analyserData.length);
  }

  const idleBob = 0.05 * Math.sin(timestampMs / 500);
  const radius = 60 + Math.max(amplitude * 80, idleBob * 60);

  ctx2d.beginPath();
  ctx2d.arc(canvas.width / 2, canvas.height / 2, radius, 0, Math.PI * 2);
  const gradient = ctx2d.createRadialGradient(
    canvas.width / 2,
    canvas.height / 2,
    radius * 0.2,
    canvas.width / 2,
    canvas.height / 2,
    radius
  );
  gradient.addColorStop(0, "#6ee7ff");
  gradient.addColorStop(1, "#1e3a8a");
  ctx2d.fillStyle = gradient;
  ctx2d.fill();

  requestAnimationFrame(drawOrb);
}

canvas = document.getElementById("orbCanvas");
statusEl = document.getElementById("status");
answerSourceEl = document.getElementById("answerSource");
transcriptEl = document.getElementById("transcript");
formEl = document.getElementById("promptForm");
inputEl = document.getElementById("promptInput");
askBtn = document.getElementById("askBtn");
stopBtn = document.getElementById("stopBtn");
clearBtn = document.getElementById("clearBtn");
micBtn = document.getElementById("micBtn");

formEl.addEventListener("submit", handleSubmit);
stopBtn.addEventListener("click", handleStop);
clearBtn.addEventListener("click", handleClear);
micBtn.addEventListener("click", handleMic);
requestAnimationFrame(drawOrb);
