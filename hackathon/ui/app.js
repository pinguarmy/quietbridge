const $ = id => document.getElementById(id);
let state = null, token = '', actionChain = Promise.resolve(), pointerHeld = false, keyHeld = false;
let speaking = false, pendingActions = 0, renderedRevision = null, initialized = false;
const reasons = {
  lip_only: 'Lip reading has no calibrated confidence score. Check the words and tap to confirm.',
  sign_only: 'A supported, stable gesture maps to this fixed phrase. Press Speak when you are ready.',
  agreement: 'The gesture matches a lip candidate. Check the complete sentence before speaking.',
  conflict: 'The inputs disagree or contain several concepts. Choose the message you intended.'
};
function requestAction(name, data = {}) {
  pendingActions++;
  actionChain = actionChain.then(async () => {
    const res = await fetch('/api/action/' + name, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Lipflow-Token': token}, body: JSON.stringify(data)});
    const result = await res.json();
    if (!res.ok) throw new Error(result.error || 'Action failed');
    render(result); $('error').textContent = '';
  }).catch(e => { $('error').textContent = e.message; }).finally(() => { pendingActions--; });
  return actionChain;
}
function render(s) {
  if (state && s.revision < state.revision) return;
  state = s; if (s.token) token = s.token;
  $('rehearsal-banner').hidden = !s.rehearsal; $('rehearsal-tools').hidden = !s.rehearsal;
  $('preview-status').textContent = s.rehearsal ? 'SYNTHETIC INPUTS' : s.camera === 'ready' ? 'LIVE · LOCAL CAMERA' : 'CAMERA ' + s.camera.toUpperCase();
  $('camera-message').textContent = s.rehearsal ? 'Rehearsal mode — camera is off' : s.camera_error || 'Waiting for camera frames…';
  $('lip-status').textContent = s.lips; $('hand-status').textContent = s.signs;
  $('last-sign').textContent = s.last_sign ? s.last_sign.label + ' · ' + s.last_sign.score.toFixed(2) : '—';
  $('component-errors').textContent = [s.camera_error, s.lip_error, s.sign_error].filter(Boolean).join('\n');
  document.querySelectorAll('[data-mode]').forEach(b => {b.classList.toggle('active', b.dataset.mode === s.mode); b.setAttribute('aria-pressed', String(b.dataset.mode === s.mode)); b.disabled = s.phase !== 'idle';});
  const recordReady = !s.rehearsal && s.lips === 'ready' && s.camera === 'ready' && s.mode !== 'signs';
  $('record').disabled = !recordReady || !['idle', 'recording'].includes(s.phase);
  $('record').classList.toggle('listening', s.phase === 'recording');
  $('record-label').textContent = ({recording:'Mouth your words… release to finish', tail:'Finishing your last word…', decoding:'Reading your lips…'})[s.phase] || 'Hold to mouth words';
  $('cancel').hidden = s.phase === 'idle';
  $('record-hint').textContent = s.rehearsal ? 'Use the synthetic input controls below to test decisions.' : s.mode === 'signs' ? 'Show a supported gesture. Keep your hands in view.' : 'Hold the button or Space for 2–4 seconds. Release to decode. Maximum 10 seconds.';
  const d = s.decision, needsReview = d.status === 'confirmation_needed';
  $('decision-status').textContent = needsReview ? 'Review & confirm' : d.status === 'accepted' ? 'Ready to speak' : s.phase === 'idle' ? 'Ready when you are' : s.phase;
  $('decision-status').classList.toggle('review', needsReview);
  $('message').textContent = d.text || (d.reason === 'conflict' ? 'Choose your message below.' : 'What would you like to say?');
  $('message').classList.toggle('empty', !d.text);
  $('reason').textContent = d.status === 'unavailable' ? 'Your camera stays on this Mac. Nothing is spoken automatically.' : reasons[d.reason];
  if (renderedRevision !== s.revision || !needsReview) {
    $('alternatives').replaceChildren();
    if (needsReview) s.alternatives.forEach(text => {
      const button = document.createElement('button'); button.className = 'alternative';
      const message = document.createElement('span'); message.textContent = text;
      const action = document.createElement('span'); action.textContent = 'Confirm ↗';
      button.append(message, action); const revision = s.revision;
      button.onclick = () => requestAction('confirm', {text, revision});
      $('alternatives').append(button);
    });
    renderedRevision = s.revision;
  }
  $('speak').disabled = d.status !== 'accepted' || !d.text || !s.speech_available || speaking;
  $('last-spoken').textContent = s.last_spoken ? 'Last spoken: ' + s.last_spoken : (!s.speech_available ? 'Native speech is unavailable on this machine.' : '');
  const values = [s.metrics.release_to_text_ms, s.metrics.decode_ms, s.metrics.fusion_ms];
  document.querySelectorAll('.metrics strong').forEach((el, i) => {el.textContent = values[i] === undefined ? '—' : values[i] + ' ms';});
  if (!initialized) {
    $('labels').replaceChildren();
    Object.entries(s.vocabulary).forEach(([label, entry]) => {
      const el = document.createElement('span'); el.textContent = label + (entry.verified ? '' : ' · pending'); el.className = entry.verified ? '' : 'pending'; $('labels').append(el);
      const opt = document.createElement('option'); opt.value = label; opt.textContent = label; $('test-label').append(opt);
    });
    initialized = true;
  }
}
async function poll() {
  try {
    const response = await fetch('/api/state'); if (!response.ok) throw new Error('Local server unavailable');
    const result = await response.json(); if (!pendingActions) render(result);
  } catch (e) { $('error').textContent = 'Connection lost. Check that the local Python server is running.'; $('speak').disabled = true; $('record').disabled = true; }
  setTimeout(poll, 180);
}
async function preview() {
  if (state && !state.rehearsal && state.camera === 'ready') {
    try {
      const res = await fetch('/api/frame'); if (res.ok) {
        const url = URL.createObjectURL(await res.blob()); const img = $('camera'), old = img.dataset.url;
        img.src = url; img.dataset.url = url; img.hidden = false; $('camera-placeholder').hidden = true;
        if (old) URL.revokeObjectURL(old);
      }
    } catch (_) {}
  } else { $('camera').hidden = true; $('camera-placeholder').hidden = false; }
  setTimeout(preview, 100);
}
document.querySelectorAll('[data-mode]').forEach(b => b.onclick = () => requestAction('mode', {mode:b.dataset.mode}));
$('record').addEventListener('pointerdown', e => {if ($('record').disabled) return; e.preventDefault(); pointerHeld = true; $('record').setPointerCapture(e.pointerId); requestAction('start');});
function releasePointer() {if (pointerHeld) {pointerHeld = false; requestAction('stop');}}
$('record').addEventListener('pointerup', releasePointer); $('record').addEventListener('pointercancel', releasePointer); $('record').addEventListener('lostpointercapture', releasePointer);
document.addEventListener('keydown', e => {
  if (['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName) || (e.target.tagName === 'BUTTON' && e.target !== $('record'))) return;
  if (e.code === 'Space' && !e.repeat && !$('record').disabled) {e.preventDefault(); keyHeld = true; requestAction('start');}
  if (e.code === 'Escape') {pointerHeld = keyHeld = false; requestAction('cancel');}
});
document.addEventListener('keyup', e => {if (e.code === 'Space' && keyHeld) {e.preventDefault(); keyHeld = false; requestAction('stop');}});
window.addEventListener('blur', () => {if (keyHeld || pointerHeld) {keyHeld = pointerHeld = false; requestAction('stop');}});
$('cancel').onclick = () => {keyHeld = pointerHeld = false; requestAction('cancel');};
$('speak').onclick = async () => {if (!state || speaking) return; speaking = true; $('speak').disabled = true; await requestAction('speak', {revision:state.revision}); speaking = false;};
$('test-submit').onclick = () => requestAction('rehearse', {text:$('test-text').value, label:$('test-label').value});
poll(); preview();
