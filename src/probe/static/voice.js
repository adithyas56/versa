/* Versa — real-time voice via LiveKit (VERSA voice-conversation context).
 *
 * The browser joins a LiveKit room, publishes the mic, and plays the agent's
 * streamed audio. The server-side voice agent does STT, calls the existing
 * Versa pipeline (Moss + Gemini), and streams TTS back. Barge-in is native.
 * NO browser SpeechRecognition / speechSynthesis is used for the conversation. */

const LK = window.LivekitClient;
const $ = (id) => document.getElementById(id);

const state = { room: null, status: "IDLE", connecting: false };

const STATUS_LABEL = {
  IDLE: "Tap to start",
  CONNECTING: "Connecting…",
  READY: "Connected — go ahead",
  LISTENING: "Listening…",
  THINKING: "Thinking…",
  RETRIEVING: "Finding your memory…",
  SPEAKING: "Versa is speaking…",
  ERROR: "Something went wrong",
};

function setStatus(s) {
  state.status = s;
  $("statusText").textContent = STATUS_LABEL[s] || s;
  $("statusDot").className = "status-dot " + s.toLowerCase();
  const orb = $("btnConnect");
  orb.classList.remove("recording", "thinking", "speaking", "connecting");
  if (s === "LISTENING") orb.classList.add("recording");
  else if (s === "THINKING" || s === "RETRIEVING") orb.classList.add("thinking");
  else if (s === "SPEAKING") orb.classList.add("speaking");
  else if (s === "CONNECTING") orb.classList.add("connecting");
  $("orbIcon").textContent =
    s === "SPEAKING" ? "🔊" : s === "LISTENING" ? "◉" : s === "CONNECTING" ? "…" : "🎙";
}
function showError(msg) {
  const el = $("err");
  if (!msg) { el.hidden = true; el.textContent = ""; return; }
  el.hidden = false; el.textContent = msg;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function clearHero() { const h = $("convo").querySelector(".empty-hero"); if (h) h.remove(); }

/* transcript rendering (best-effort; conversation works even if this is quiet) */
const _turns = {}; // id -> element, to update interim → final
function renderTranscript(id, who, text, final) {
  clearHero();
  const convo = $("convo");
  let el = _turns[id];
  if (!el) {
    el = document.createElement("div");
    el.className = "turn " + who;
    el.innerHTML = `<div class="who">${who === "student" ? "You" : "Versa"}</div><div class="bubble"></div>`;
    convo.appendChild(el);
    _turns[id] = el;
  }
  el.querySelector(".bubble").textContent = text;
  el.style.opacity = final ? "1" : "0.7";
  convo.scrollTop = convo.scrollHeight;
}

function renderMoss(retrieval) {
  const moss = $("mossPanel");
  if (!retrieval) { return; }
  const mems = retrieval.memories || [];
  const engineLabel = retrieval.fell_back ? "fallback" : retrieval.engine;
  moss.innerHTML =
    `<div class="moss-stat">` +
    `<div><span class="k">Engine</span><b>${escapeHtml(String(engineLabel))}</b></div>` +
    `<div><span class="k">Latency</span><b>${Number(retrieval.latency_ms).toFixed(2)} ms</b></div>` +
    `<div><span class="k">Retrieved</span><b>${retrieval.result_count}</b></div>` +
    `<div><span class="k">Index</span><b>${retrieval.local_index ? "local" : "remote"}</b></div>` +
    `</div>` +
    mems.slice(0, 3).map((m) => `<div class="moss-mem">${escapeHtml(m.statement || "")}</div>`).join("");
}

function setBadges(connected) {
  $("engineBadges").innerHTML =
    `<span class="badge ${connected ? "on" : ""}"><span class="d"></span>LiveKit ${connected ? "· live" : ""}</span>` +
    `<span class="badge on"><span class="d"></span>Moss</span>` +
    `<span class="badge on"><span class="d"></span>Gemini</span>`;
}

function mapAgentState(st) {
  // livekit-agents publishes lk.agent.state: initializing|listening|thinking|speaking|idle
  if (st === "listening") setStatus("LISTENING");
  else if (st === "thinking") setStatus("THINKING");
  else if (st === "speaking") setStatus("SPEAKING");
  else if (st === "idle" || st === "initializing") setStatus("READY");
}

async function connect() {
  if (state.connecting || state.room) return;
  if (!LK) { showError("LiveKit failed to load. Check your connection and refresh."); return; }
  state.connecting = true; showError(""); setStatus("CONNECTING");
  const learner = ($("learnerInput").value || "demo-learner-a").trim();
  try {
    const res = await fetch("/api/livekit/token", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ learner }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "could not get token");

    const room = new LK.Room({ adaptiveStream: true, dynacast: true });
    state.room = room;

    room.on(LK.RoomEvent.TrackSubscribed, (track) => {
      if (track.kind === LK.Track.Kind.Audio) {
        const el = track.attach();
        el.autoplay = true;
        $("audioContainer").appendChild(el);
      }
    });
    room.on(LK.RoomEvent.ParticipantAttributesChanged, (_changed, participant) => {
      const st = participant.attributes && participant.attributes["lk.agent.state"];
      if (st) mapAgentState(st);
    });
    room.on(LK.RoomEvent.DataReceived, (payload, _p, _kind, topic) => {
      if (topic === "versa.telemetry") {
        try { renderMoss(JSON.parse(new TextDecoder().decode(payload)).retrieval); } catch (_) {}
      }
    });
    // transcripts (best-effort; API name varies by SDK version)
    try {
      room.on(LK.RoomEvent.TranscriptionReceived, (segments, participant) => {
        const who = participant && participant.isLocal ? "student" : "versa";
        for (const seg of segments) renderTranscript(seg.id, who, seg.text, seg.final);
      });
    } catch (_) {}
    room.on(LK.RoomEvent.Disconnected, () => teardown());

    await room.connect(data.url, data.token);
    await room.localParticipant.setMicrophoneEnabled(true);
    setBadges(true);
    setStatus("READY");
    $("btnEnd").hidden = false;
  } catch (e) {
    showError("Couldn't start: " + e.message);
    setStatus("ERROR");
    teardown();
  } finally {
    state.connecting = false;
  }
}

function teardown() {
  if (state.room) { try { state.room.disconnect(); } catch (_) {} }
  state.room = null;
  $("audioContainer").innerHTML = "";
  $("btnEnd").hidden = true;
  setBadges(false);
  if (state.status !== "ERROR") setStatus("IDLE");
}

$("btnConnect").onclick = () => { state.room ? teardown() : connect(); };
$("btnEnd").onclick = teardown;
setBadges(false);
setStatus("IDLE");
if (!LK) showError("LiveKit client didn't load (CDN blocked?). Refresh or check your network.");
