// Test page for the HeyGen LiveAvatar integration. LiveKit is used here
// ONLY as a read-only viewer for the video/audio HeyGen renders -- this page
// never sends audio into the room. The actual lip-sync audio is pushed by
// the BACKEND directly to HeyGen's WebSocket (see app/services/heygen_streaming.py);
// the browser never sees the HeyGen API key or that WebSocket URL, only the
// livekit_url/livekit_client_token that arrive in "session" frames.
//
// ONE /ws/avatar connection now lasts the whole tab, not one per Ask. HeyGen
// session setup (~4.4s) and the TTS socket (~2.3s) were previously paid on
// EVERY question; the backend now builds both once and reuses them, only
// rebuilding after ~2 minutes of no questions (see IDLE_RELEASE_TIMEOUT_S in
// avatar_routes.py). A "session" frame can therefore arrive more than once on
// the same socket -- the first time, and again any time the backend rebuilds
// after an idle release -- so this page always re-joins LiveKit and replies
// "ready" whenever one arrives, rather than assuming it only happens once.
import { Room, RoomEvent } from "https://cdn.jsdelivr.net/npm/livekit-client/+esm";

let room = null;
let questionSentAt = null;
let firstTextReceived = false;
// Conversation thread for memory. Minted once per page load and sent with
// every question; the server echoes back whichever thread it used, and only
// ANSWER_SOURCE=agent actually consults it. Deliberately NOT tied to the
// WebSocket -- a reconnect (Stop, network blip) must not wipe the
// conversation. Same contract app.js already uses for /ws/voice.
let threadId = crypto.randomUUID();
// Time-to-first-SOUND, measured here rather than server-side. The backend's
// "time to first audio" fires when audio leaves OpenAI for HeyGen, which is
// several seconds before anything is audible -- HeyGen still has to render and
// LiveKit still has to deliver. Only the browser can see the real moment.
let firstSoundReported = false;
let soundWatchRaf = null;
let soundAudioCtx = null;

// Incremented on every new Ask AND on Stop, so message/event handlers from a
// turn that's been superseded or explicitly stopped can recognize they're
// stale and no-op instead of touching the UI for a turn that's no longer
// current -- same pattern as static/app.js's turnId.
let currentTurn = 0;
// The persistent /ws/avatar connection for this tab, opened lazily on the
// first Ask and reused for every question after that -- see the module
// docstring above. null whenever no connection is open.
let avatarWs = null;
// { myTurn, resolve, reject } for whichever turn is currently awaiting the
// server's "done"/"error" for its question. Only one turn is ever in flight
// at a time (Ask is disabled while busy), so a single slot is enough -- no
// queue of pending turns is needed.
let pendingTurn = null;

let videoEl, statusEl, answerSourceEl, latencyEl, transcriptEl, formEl, inputEl, askBtn, stopBtn, clearBtn;

function getWsUrl(path) {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${window.location.host}${path}`;
}

function setStatus(text, kind = "idle") {
  statusEl.textContent = text;
  statusEl.className = `status status--${kind}`;
}

function setBusy(isBusy) {
  askBtn.disabled = isBusy;
  inputEl.disabled = isBusy;
  stopBtn.hidden = !isBusy;
}

function appendTurn(question) {
  const prefix = transcriptEl.textContent ? "\n\n" : "";
  transcriptEl.textContent += `${prefix}You: ${question}\nAssistant: `;
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function appendSentence(text) {
  transcriptEl.textContent += (transcriptEl.textContent.endsWith(": ") ? "" : " ") + text;
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function showAnswerSource(msg) {
  if (msg.answer_source === "rag" || msg.answer_source === "agent") {
    const verdict = msg.grounded ? "grounded ✓" : "NOT grounded — abstaining";
    answerSourceEl.textContent = `Source: ${msg.answer_source.toUpperCase()} · ${verdict}`;
    answerSourceEl.className = `answer-source answer-source--${msg.grounded ? "rag" : "blocked"}`;
  } else {
    answerSourceEl.textContent = "Source: GPT — direct LLM, NO retrieval";
    answerSourceEl.className = "answer-source answer-source--gpt";
  }
  answerSourceEl.hidden = false;
}

// Watches the avatar's audio track and reports the first moment it actually
// makes a sound. An RMS threshold rather than the <video> "playing" event,
// because the track starts flowing (silent, idle avatar) the instant we join
// the room -- "playing" would fire seconds before a single word is spoken.
function watchForFirstSound(mediaStreamTrack) {
  if (firstSoundReported || soundWatchRaf !== null) return;
  try {
    soundAudioCtx = soundAudioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const source = soundAudioCtx.createMediaStreamSource(new MediaStream([mediaStreamTrack]));
    const analyser = soundAudioCtx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    const data = new Uint8Array(analyser.fftSize);

    const tick = () => {
      analyser.getByteTimeDomainData(data);
      let sumSquares = 0;
      for (const v of data) {
        const d = (v - 128) / 128;
        sumSquares += d * d;
      }
      const amplitude = Math.sqrt(sumSquares / data.length);
      if (amplitude > 0.02) {
        reportFirstSound();
        return;
      }
      soundWatchRaf = requestAnimationFrame(tick);
    };
    soundWatchRaf = requestAnimationFrame(tick);
  } catch (_err) {
    // Audio analysis is instrumentation only -- never let it break the demo.
  }
}

function reportFirstSound() {
  if (firstSoundReported) return;
  firstSoundReported = true;
  stopSoundWatch();
  const ms = Math.round(performance.now() - questionSentAt);
  latencyEl.textContent = `${latencyEl.textContent}  ·  time to first SOUND: ${ms}ms (measured in the browser)`;
}

function stopSoundWatch() {
  if (soundWatchRaf !== null) {
    cancelAnimationFrame(soundWatchRaf);
    soundWatchRaf = null;
  }
}

async function connectLiveKit(livekitUrl, livekitClientToken) {
  if (room) {
    await room.disconnect();
    room = null;
  }
  room = new Room();
  room.on(RoomEvent.TrackSubscribed, (track) => {
    // Renders whatever HeyGen publishes (video + its audio track carrying
    // our TTS audio, lip-synced) straight into the <video> element.
    track.attach(videoEl);
    if (track.kind === "audio" && track.mediaStreamTrack) {
      watchForFirstSound(track.mediaStreamTrack);
    }
  });
  await room.connect(livekitUrl, livekitClientToken);
}

// Resolves the CURRENT pendingTurn (if any) and clears the slot. Used from
// every place a turn can end: done, error, or the socket closing under it.
function settlePendingTurn(fn) {
  const turn = pendingTurn;
  if (!turn) return;
  pendingTurn = null;
  fn(turn);
}

async function handleAvatarMessage(event) {
  if (!pendingTurn) return; // stray message with no turn waiting on it
  const { myTurn } = pendingTurn;
  if (myTurn !== currentTurn) return; // stale -- Stop was pressed
  const msg = JSON.parse(event.data);

  if (msg.type === "session") {
    // A new HeyGen session (first one, or a rebuild after an idle release).
    // Join it, then tell the backend we can see it -- it holds the first
    // sentence of whichever turn is in flight until this reply arrives.
    setStatus("Joining avatar video…");
    try {
      await connectLiveKit(msg.livekit_url, msg.livekit_client_token);
    } catch (err) {
      setStatus(`Error joining avatar video: ${err.message}`, "error");
    }
    if (myTurn !== currentTurn || !avatarWs) return;
    avatarWs.send(JSON.stringify({ type: "ready" }));
    setStatus("Waiting for answer…");
  } else if (msg.type === "meta") {
    // Server echoes the thread it actually used -- keep reusing that one, so
    // a server-minted fallback (or anything else it decides) still gives us
    // continuity on later questions.
    if (msg.thread_id) threadId = msg.thread_id;
    showAnswerSource(msg);
  } else if (msg.type === "text") {
    if (!firstTextReceived) {
      firstTextReceived = true;
      const ttftMs = Math.round(performance.now() - questionSentAt);
      latencyEl.textContent = `Time to first text: ${ttftMs}ms`;
    }
    appendSentence(msg.text);
  } else if (msg.type === "speaking") {
    setStatus("Avatar speaking…", "speaking");
  } else if (msg.type === "done") {
    setStatus("Done — ask another question anytime", "idle");
    settlePendingTurn((turn) => turn.resolve());
  } else if (msg.type === "error") {
    setStatus(`Error: ${msg.detail}`, "error");
    settlePendingTurn((turn) => turn.reject(new Error(msg.detail)));
  }
}

function connectAvatarSocket() {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(getWsUrl("/ws/avatar"));
    ws.onopen = () => resolve(ws);
    ws.onerror = () => reject(new Error("failed to connect to the avatar"));
    ws.onmessage = handleAvatarMessage;
    ws.onclose = () => {
      if (avatarWs === ws) avatarWs = null;
      // A turn still waiting when the socket drops (server restart, network
      // loss) needs to be released, not left hanging forever.
      settlePendingTurn((turn) => turn.reject(new Error("avatar connection closed")));
    };
    avatarWs = ws;
  });
}

async function ensureAvatarSocket() {
  if (avatarWs && avatarWs.readyState === WebSocket.OPEN) return avatarWs;
  return connectAvatarSocket();
}

function askOverWebSocket(question, myTurn) {
  return new Promise((resolve, reject) => {
    ensureAvatarSocket()
      .then((ws) => {
        if (myTurn !== currentTurn) {
          resolve();
          return;
        }
        questionSentAt = performance.now();
        firstTextReceived = false;
        pendingTurn = { myTurn, resolve, reject };
        ws.send(JSON.stringify({ question, thread_id: threadId }));
        setStatus("Thinking…");
      })
      .catch(reject);
  });
}

async function handleSubmit(event) {
  event.preventDefault();
  const question = inputEl.value.trim();
  if (!question) return;

  currentTurn += 1;
  const myTurn = currentTurn;

  setBusy(true);
  answerSourceEl.hidden = true;
  latencyEl.textContent = "";
  firstSoundReported = false;
  stopSoundWatch();
  appendTurn(question);
  inputEl.value = "";

  try {
    // Reuses the existing avatar connection (and its HeyGen session/TTS
    // socket) if one is already open from an earlier question on this tab;
    // opens one only if this is the first Ask, or the previous connection
    // was closed (Stop, or the backend's idle release -- see
    // IDLE_RELEASE_TIMEOUT_S in avatar_routes.py).
    await askOverWebSocket(question, myTurn);
  } catch (err) {
    if (myTurn === currentTurn) {
      setStatus(`Error: ${err.message}`, "error");
    }
  } finally {
    if (myTurn === currentTurn) {
      setBusy(false);
      inputEl.focus();
    }
  }
}

async function handleStop() {
  currentTurn += 1; // invalidate any in-flight handlers for the turn being stopped
  stopSoundWatch();
  // Closes the whole avatar connection, not just the in-flight turn --
  // simplest and safest option, at the cost of discarding the warmed-up
  // HeyGen session: the next Ask pays full setup again. Stopping just the
  // current turn while keeping the connection alive is possible but wasn't
  // built here; flagged as a possible refinement, not done speculatively.
  if (avatarWs) {
    avatarWs.close();
    avatarWs = null;
  }
  pendingTurn = null;
  if (room) {
    await room.disconnect();
    room = null;
  }
  setStatus("Stopped — press Ask to start a new question", "idle");
  setBusy(false);
}

function handleClear() {
  transcriptEl.textContent = "";
  answerSourceEl.hidden = true;
  answerSourceEl.textContent = "";
  latencyEl.textContent = "";
  // A real new conversation, not just a visual wipe: a fresh thread means the
  // server starts from empty history rather than silently carrying the old
  // conversation into questions the user thinks are a clean slate.
  threadId = crypto.randomUUID();
}

videoEl = document.getElementById("avatarVideo");
statusEl = document.getElementById("status");
answerSourceEl = document.getElementById("answerSource");
latencyEl = document.getElementById("latency");
transcriptEl = document.getElementById("transcript");
formEl = document.getElementById("promptForm");
inputEl = document.getElementById("promptInput");
askBtn = document.getElementById("askBtn");
stopBtn = document.getElementById("stopBtn");
clearBtn = document.getElementById("clearBtn");

formEl.addEventListener("submit", handleSubmit);
stopBtn.addEventListener("click", handleStop);
clearBtn.addEventListener("click", handleClear);
