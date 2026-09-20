/* probe — instrument demo client (locate).
 *
 * Standalone page, no relation to app.js / the chat SPA (no mode
 * selector: presenting an instrument is never triggered from a
 * SessionLoop turn). Captures the raw event stream the same way any
 * future primitive's client would: element id + coordinates per
 * click, elapsed_ms from presentation on every event, and an explicit
 * abandon signal when the tab is hidden/closed before a submit ever
 * fires. Interpretation happens entirely server-side
 * (interpret_instrument) — this file never decides supports/
 * contradicts/uninformative, it only reports what happened.
 */

const $ = (id) => document.getElementById(id);

const state = {
  instrumentId: null,
  presentedAtMs: null,
  seq: 0,
  chosenElement: null,
  finalized: false,
  primitive: 'locate',
  payloadField: 'clicked_element',
};

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  return res.json();
}

async function getJSON(url) {
  const res = await fetch(url);
  return res.json();
}

async function refreshInspector() {
  if (state.instrumentId === null) return;
  const data = await getJSON(`/api/instrument/${state.instrumentId}/inspect`);
  if (data.error) return;
  const claim = data.target_claim;
  $('inspClaim').textContent = claim ? `${claim.kind}: ${claim.skill || claim.value}` : '(none yet)';
  $('inspConfidence').textContent = claim ? claim.confidence.toFixed(3) : '—';
  $('inspEvidenceCount').textContent = claim ? claim.evidence_count : '—';
  $('inspStatus').textContent = data.instrument.abandoned
    ? 'abandoned'
    : (data.instrument.completed_at ? 'completed' : 'in progress');
  $('inspEvents').textContent = data.events
    .map((e) => `[${e.seq}] ${e.event_type} @${e.elapsed_ms}ms ${JSON.stringify(e.payload)}`)
    .join('\n') || '(no events yet)';
  $('inspector').classList.add('show');
}

function elapsedMs() {
  return Math.round(performance.now() - state.presentedAtMs);
}

function nextSeq() {
  return state.seq++;
}

async function emit(eventType, payload) {
  if (state.instrumentId === null) return;
  await postJSON(`/api/instrument/${state.instrumentId}/event`, {
    seq: nextSeq(),
    event_type: eventType,
    payload: payload || {},
    elapsed_ms: elapsedMs(),
  });
}

async function present() {
  const learner = $('learnerInput').value.trim();
  const primitive = $('primitiveSelect').value;
  if (!learner) return;
  $('presentBtn').disabled = true;
  const data = await postJSON('/api/instrument/present', { learner, primitive });
  if (data.error) {
    alert(data.error);
    $('presentBtn').disabled = false;
    return;
  }
  state.instrumentId = data.instrument_id;
  state.presentedAtMs = performance.now();
  state.seq = 0;
  state.chosenElement = null;
  state.finalized = false;
  state.primitive = data.primitive;
  // Both hand-authored contracts key their click predicate on a
  // different payload field ("clicked_element" for locate,
  // "chosen_option" for predict) -- interpret_instrument doesn't care
  // which name is used, only that the contract and the client agree.
  state.payloadField = data.primitive === 'predict' ? 'chosen_option' : 'clicked_element';

  $('setupArea').style.display = 'none';
  $('promptText').textContent = data.spec.prompt;

  const machineryBlock = $('machineryBlock');
  const clickables = data.spec.steps || data.spec.options;
  if (data.spec.machinery) {
    machineryBlock.textContent = data.spec.machinery.join('\n') +
      (data.spec.question ? `\n\n${data.spec.question}` : '');
    machineryBlock.style.display = 'block';
  } else {
    machineryBlock.style.display = 'none';
  }

  const list = $('stepsList');
  list.innerHTML = '';
  for (const item of clickables) {
    const div = document.createElement('div');
    div.className = 'step';
    div.textContent = item.text;
    div.dataset.elementId = item.id;
    div.addEventListener('click', (evt) => onStepClick(item.id, evt));
    list.appendChild(div);
  }
  $('resultBox').className = 'result';
  $('submitBtn').disabled = true;
  $('instrumentArea').classList.add('show');

  await emit('start', {});
  await refreshInspector();
}

function onStepClick(elementId, evt) {
  if (state.finalized) return;
  state.chosenElement = elementId;
  for (const el of $('stepsList').children) {
    el.classList.toggle('chosen', el.dataset.elementId === elementId);
  }
  $('submitBtn').disabled = false;
  // Every click, including ones later changed, is its own event --
  // the server-side contract reads the whole stream, not just the
  // final click.
  emit('click', { [state.payloadField]: elementId, x: evt.clientX, y: evt.clientY })
    .then(refreshInspector);
}

async function submit() {
  if (state.chosenElement === null || state.finalized) return;
  state.finalized = true;
  $('submitBtn').disabled = true;
  await emit('submit', { [state.payloadField]: state.chosenElement });
  const result = await postJSON(`/api/instrument/${state.instrumentId}/finalize`, {});
  const box = $('resultBox');
  box.className = `result show ${result.outcome}`;
  box.textContent = `outcome: ${result.outcome} — evidence written: ${result.evidence_written}`;
  await refreshInspector();
}

async function onAbandon() {
  if (state.instrumentId === null || state.finalized) return;
  state.finalized = true; // best-effort, page may not survive long enough for finalize to matter
  await emit('abandon', {});
  await postJSON(`/api/instrument/${state.instrumentId}/finalize`, {});
}

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') onAbandon();
});
window.addEventListener('pagehide', onAbandon);

$('presentBtn').addEventListener('click', present);
$('submitBtn').addEventListener('click', submit);
$('refreshInspectorBtn').addEventListener('click', refreshInspector);
