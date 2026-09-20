/* Versa voice tutor — browser voice (speech-in via Web Speech API,
 * speech-out via the browser's built-in synthesis, with automatic upgrade
 * to server audio when available). Voice is only an I/O surface: every
 * turn runs the existing Versa pipeline server-side (POST /api/voice/turn).
 * No API keys touch this file. */

const $ = (id) => document.getElementById(id);

const state = { learner: "demo-learner-a", sessionId: null, status: "IDLE", busy: false };

const STATUS_LABEL = {
  IDLE: "Tap to speak",
  LISTENING: "Listening…",
  RETRIEVING: "Finding what Versa remembers…",
  THINKING: "Thinking…",
  SPEAKING: "Speaking…",
  ERROR: "Something went wrong",
};

/* ---------- speech recognition ---------- */
function createSpeech() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return null;
  const rec = new SR();
  rec.continuous = false; rec.interimResults = true; rec.lang = "en-IN";
  const api = { onInterim: null, onFinal: null, onError: null, onEnd: null };
  rec.onresult = (e) => {
    let interim = "", finalText = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const t = e.results[i][0].transcript;
      if (e.results[i].isFinal) finalText += t; else interim += t;
    }
    if (interim && api.onInterim) api.onInterim(interim);
    if (finalText && api.onFinal) api.onFinal(finalText.trim());
  };
  rec.onerror = (e) => { if (api.onError) api.onError(e.error || "speech error"); };
  rec.onend = () => { if (api.onEnd) api.onEnd(); };
  api.start = () => { try { rec.start(); } catch (_) {} };
  api.stop = () => { try { rec.stop(); } catch (_) {} };
  return api;
}
const speech = createSpeech();

/* ---------- helpers ---------- */
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
/* Make a tutor answer readable in a chat bubble: drop code fences, math
 * delimiters, and markdown emphasis so the transcript reads like speech. */
function cleanForDisplay(text) {
  let t = String(text || "");
  t = t.replace(/```[\s\S]*?```/g, " ");
  t = t.replace(/\$+([^$]*)\$+/g, "$1");          // $x$ -> x
  t = t.replace(/\\frac\{([^}]*)\}\{([^}]*)\}/g, "($1)/($2)");
  t = t.replace(/\\[a-zA-Z]+/g, "");               // stray latex commands
  t = t.replace(/[*_`#>]/g, "");
  t = t.replace(/[ \t]{2,}/g, " ");
  return t.trim();
}

/* ---------- state / orb ---------- */
function setStatus(s) {
  state.status = s;
  $("statusText").textContent = STATUS_LABEL[s] || s;
  $("statusDot").className = "status-dot " + s.toLowerCase();
  const orb = $("btnMic");
  orb.classList.remove("recording", "thinking", "speaking");
  if (s === "LISTENING") orb.classList.add("recording");
  else if (s === "THINKING" || s === "RETRIEVING") orb.classList.add("thinking");
  else if (s === "SPEAKING") orb.classList.add("speaking");
  $("orbIcon").textContent = s === "LISTENING" ? "◉" : s === "SPEAKING" ? "🔊" : "🎙";
}
function showError(msg) {
  const el = $("err");
  if (!msg) { el.hidden = true; el.textContent = ""; return; }
  el.hidden = false; el.textContent = msg;
}

function clearHero() { const h = $("convo").querySelector(".empty-hero"); if (h) h.remove(); }
function addTurn(who, text) {
  clearHero();
  const convo = $("convo");
  const div = document.createElement("div");
  div.className = "turn " + who;
  const body = who === "versa" ? cleanForDisplay(text) : text;
  div.innerHTML = `<div class="who">${who === "student" ? "You" : "Versa"}</div>` +
    `<div class="bubble">${escapeHtml(body)}</div>`;
  convo.appendChild(div);
  convo.scrollTop = convo.scrollHeight;
}

function renderOptions(options) {
  const box = $("options");
  box.innerHTML = "";
  if (!options || !options.length) { box.hidden = true; return; }
  box.hidden = false;
  options.forEach((o) => {
    const b = document.createElement("button");
    b.className = "opt"; b.textContent = o.text;
    b.onclick = () => sendTurn(o.text, o.id);
    box.appendChild(b);
  });
}

function renderMemory(retrieval) {
  const list = $("memoryList"), moss = $("mossPanel");
  if (!retrieval) {
    list.innerHTML = '<div class="muted">No memory retrieved yet.</div>';
    moss.innerHTML = '<div class="muted">Awaiting first turn…</div>';
    return;
  }
  const mems = retrieval.memories || [];
  list.innerHTML = mems.length
    ? mems.map((m) => `<div class="mem-item"><span class="tick">✓</span><span>${escapeHtml(cleanForDisplay(m.statement))}</span></div>`).join("")
    : '<div class="muted">No learner memory matched this turn.</div>';
  const engineLabel = retrieval.fell_back ? "fallback" : retrieval.engine;
  moss.innerHTML =
    `<div class="moss-stat">` +
    `<div><span class="k">Engine</span><b>${escapeHtml(engineLabel)}</b></div>` +
    `<div><span class="k">Latency</span><b>${Number(retrieval.latency_ms).toFixed(2)} ms</b></div>` +
    `<div><span class="k">Retrieved</span><b>${retrieval.result_count}</b></div>` +
    `<div><span class="k">Index</span><b>${retrieval.local_index ? "local" : "remote"}</b></div>` +
    `</div>` +
    mems.slice(0, 3).map((m) => `<div class="moss-mem">${escapeHtml(cleanForDisplay(m.statement))}</div>`).join("");
}

function setEngineBadges(retrieval, voiceEngine) {
  const el = $("engineBadges");
  const mossReal = retrieval && retrieval.engine === "moss";
  const ttsOn = voiceEngine === "elevenlabs" || voiceEngine === "browser";
  const b = [];
  b.push(`<span class="badge on"><span class="d"></span>Gemini</span>`);
  b.push(`<span class="badge ${mossReal ? "on" : ""}"><span class="d"></span>Moss ${retrieval ? "· " + retrieval.engine : ""}</span>`);
  b.push(`<span class="badge ${ttsOn ? "on" : ""}"><span class="d"></span>Voice ${voiceEngine ? "· " + voiceEngine : ""}</span>`);
  el.innerHTML = b.join("");
}

/* ---------- audio out ---------- */
let currentAudio = null;
function stopAudio() {
  try { const p = $("player"); p.pause(); p.currentTime = 0; } catch (_) {}
  currentAudio = null;
  try { if (window.speechSynthesis) window.speechSynthesis.cancel(); } catch (_) {}
}
function playAudio(b64, mime) {
  stopAudio();
  const p = $("player");
  p.src = `data:${mime || "audio/mpeg"};base64,${b64}`;
  currentAudio = p;
  setStatus("SPEAKING");
  p.onended = () => { if (state.status === "SPEAKING") setStatus("IDLE"); };
  p.play().catch(() => { if (state.status === "SPEAKING") setStatus("IDLE"); });
}
function speakBrowser(text) {
  if (!text || !window.speechSynthesis || !window.SpeechSynthesisUtterance) return false;
  try {
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.0; u.pitch = 1.0; u.lang = "en-US";
    const voices = window.speechSynthesis.getVoices() || [];
    const en = voices.find((v) => /^en(-|_|$)/i.test(v.lang));
    if (en) u.voice = en;
    setStatus("SPEAKING");
    u.onend = () => { if (state.status === "SPEAKING") setStatus("IDLE"); };
    u.onerror = () => { if (state.status === "SPEAKING") setStatus("IDLE"); };
    window.speechSynthesis.speak(u);
    return true;
  } catch (_) { return false; }
}

/* ---------- the turn ---------- */
async function sendTurn(text, optionId) {
  text = (text || "").trim();
  if (!text || state.busy) return;
  state.busy = true; showError(""); $("interim").hidden = true;
  renderOptions([]); addTurn("student", text); setStatus("RETRIEVING");
  try {
    const res = await fetch("/api/voice/turn", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, learner: state.learner, text, option_id: optionId || null }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "turn failed");
    state.sessionId = data.session_id;
    setStatus("THINKING");
    addTurn("versa", data.message);
    renderOptions(data.branched ? data.pending_options : []);
    renderMemory(data.retrieval);
    let voiceUsed = data.voice_engine;
    if (data.audio_base64) { playAudio(data.audio_base64, data.audio_mime); }
    else { const spoke = speakBrowser(data.spoken_text || data.message); voiceUsed = spoke ? "browser" : null; if (!spoke) setStatus("IDLE"); }
    setEngineBadges(data.retrieval, voiceUsed);
  } catch (e) {
    setStatus("ERROR"); showError(e.message);
    setTimeout(() => { if (state.status === "ERROR") setStatus("IDLE"); }, 2600);
  } finally { state.busy = false; }
}

/* ---------- mic ---------- */
let recording = false;
function startListening() {
  if (!speech) { showError("Speech input needs Chrome — use the type box below."); return; }
  stopAudio(); showError(""); recording = true;
  $("interim").hidden = false; $("interim").textContent = "…";
  setStatus("LISTENING");
  speech.onInterim = (t) => { $("interim").textContent = t; };
  speech.onError = (err) => { stopListening(); if (err !== "no-speech" && err !== "aborted") showError("Mic: " + err); };
  speech.onEnd = () => { stopListening(); };
  speech.onFinal = (t) => { stopListening(); sendTurn(t); };
  speech.start();
}
function stopListening() {
  recording = false; $("interim").hidden = true;
  if (speech) speech.stop();
  if (state.status === "LISTENING") setStatus("IDLE");
}
/* Orb: barge-in when speaking, stop when recording, else start. (§33) */
$("btnMic").onclick = () => {
  if (state.status === "SPEAKING") { stopAudio(); setStatus("IDLE"); return; }
  recording ? stopListening() : startListening();
};

/* ---------- learners / reset / typed ---------- */
document.querySelectorAll(".learner-btn").forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll(".learner-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active"); state.learner = b.dataset.learner; resetSession();
  };
});
$("btnReset").onclick = resetSession;
function resetSession() {
  stopAudio(); stopListening(); state.sessionId = null;
  $("convo").innerHTML =
    '<div class="empty-hero"><div class="hero-orb"></div>' +
    '<div class="hero-line">New session.</div>' +
    '<div class="hero-sub">Tap the mic and ask a question.</div></div>';
  renderOptions([]); renderMemory(null); $("engineBadges").innerHTML = ""; setStatus("IDLE");
}
$("typedForm").onsubmit = (e) => {
  e.preventDefault();
  const v = $("typedInput").value.trim();
  if (v) { $("typedInput").value = ""; sendTurn(v); }
};

/* warm up the browser voice list (some browsers populate async) */
if (window.speechSynthesis) { try { window.speechSynthesis.getVoices(); } catch (_) {} }
setStatus("IDLE");
if (!speech) showError("Voice input needs Chrome's Web Speech API — the type box works everywhere.");
