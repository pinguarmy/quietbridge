const $ = id => document.getElementById(id);
const text = (id, value) => { const el = $(id), next = String(value ?? ''); if (el && el.textContent !== next) el.textContent = next; };
const hide = (id, value) => { if ($(id)) $(id).hidden = value; };
let state = null, token = '', actionChain = Promise.resolve(), pointerHeld = false, keyHeld = false;
let speaking = false, pendingActions = 0, renderedRevision = null, initialized = false, chosenText = null, connectionLost = false;
let messageAnimation = null;

function requestAction(name, data = {}) {
  pendingActions++;
  actionChain = actionChain.then(async () => {
    const res = await fetch('/api/action/' + name, {method:'POST', headers:{'Content-Type':'application/json', 'X-Lipflow-Token':token}, body:JSON.stringify(data)});
    const result = await res.json();
    if (!res.ok) throw new Error(result.error || 'Please try again.');
    render(result); text('error', ''); return true;
  }).catch(e => { text('error', e.message); return false; }).finally(() => { pendingActions--; });
  return actionChain;
}

function render(s) {
  if (state && s.revision < state.revision) return;
  state = s; if (s.token) token = s.token;
  hide('rehearsal-banner', !s.rehearsal); hide('preview', !s.rehearsal);
  text('camera-status', s.camera); text('lip-status', s.lips); text('hand-status', s.signs);
  text('last-sign', s.last_sign ? s.last_sign.label + ' · ' + s.last_sign.score.toFixed(2) : '—');
  hide('gesture-feedback', s.mode === 'lips');
  const lastGesture = s.last_sign;
  const gestureName = lastGesture ? (s.vocabulary[lastGesture.label]?.display_name || lastGesture.label) : '';
  text('gesture-label', lastGesture ? (s.rehearsal ? 'Synthetic gesture: ' : 'Last stable gesture: ') + gestureName : s.rehearsal ? 'Preview mode · no live recognition' : s.signs === 'ready' ? 'Gesture input ready' : s.signs === 'unavailable' ? 'Gesture input needs setup' : 'Getting gesture input ready…');
  const gestureMetrics = [];
  if (lastGesture) gestureMetrics.push('model score ' + lastGesture.score.toFixed(2));
  if (!s.rehearsal && Number.isFinite(s.metrics.hand_process_ms)) gestureMetrics.push('processing ' + s.metrics.hand_process_ms.toFixed(1) + ' ms');
  text('gesture-metrics', gestureMetrics.length ? ' · ' + gestureMetrics.join(' · ') : '');
  text('component-errors', [s.camera_error, s.lip_error, s.sign_error].filter(Boolean).join('\n'));
  const modeName = {combined:'Lips and gestures', lips:'Lips only', signs:'Gestures only'}[s.mode];
  text('mode-caption', modeName);
  if ($('test-text')) $('test-text').disabled = s.mode === 'signs';
  if ($('test-label')) $('test-label').disabled = s.mode === 'lips';
  text('mode-hint', s.phase === 'idle' ? 'Either input can work on its own.' : 'Finish or cancel your recording before changing inputs.');
  document.querySelectorAll('[data-mode]').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === s.mode);
    b.setAttribute('aria-pressed', String(b.dataset.mode === s.mode)); b.disabled = s.phase !== 'idle';
  });
  const ready = s.camera === 'ready';
  text('preview-status', s.rehearsal ? 'Preview · camera off' : ready ? 'Camera on' : s.camera === 'unavailable' ? 'Camera unavailable' : 'Getting ready…');
  text('camera-message', s.rehearsal ? 'Your camera is off in preview.' : s.camera === 'unavailable' ? 'Check camera access in Settings.' : 'Getting your camera ready…');
  const record = $('record');
  if (record) {
    hide('record', s.mode === 'signs');
    record.disabled = s.rehearsal || s.lips !== 'ready' || !ready || !['idle','recording'].includes(s.phase);
    record.classList.toggle('listening', s.phase === 'recording');
    text('record-label', ({recording:'Mouth your words… release to finish', tail:'Finishing…', decoding:'Reading your lips…'})[s.phase] || 'Hold to mouth words');
  }
  hide('cancel', s.phase === 'idle');
  text('record-hint', s.rehearsal ? 'Try an example from Settings.' : s.mode === 'signs' ? 'Show a supported gesture to the camera.' : s.lips === 'warming' ? 'Getting the lip reader ready…' : s.lips === 'unavailable' ? 'Lip reading needs attention in Settings.' : s.mode === 'combined' ? 'Hold Space or the button before mouthing and gesturing.' : 'Or hold Space, then release when you finish.');
  text('entry-hint', s.mode === 'signs' ? 'Show a supported gesture. Review, then speak.' : s.mode === 'lips' ? 'Hold the button, mouth a short sentence, then release.' : 'Show a supported gesture, or hold the button to mouth a sentence.');
  const gesturesReady = s.signs === 'ready' && Object.values(s.vocabulary).some(v => v.enabled ?? v.verified);
  if (!s.rehearsal && !gesturesReady && s.mode !== 'lips') {
    text('entry-hint', s.mode === 'signs' ? 'Gestures need setup. Choose another input in Settings.' : 'Hold the button to mouth a short sentence.');
    text('record-hint', s.mode === 'signs' ? 'Open Settings to check gesture input.' : 'Gesture input is being set up. You can still use your lips once ready.');
  }
  const d = s.decision, review = d.status === 'confirmation_needed';
  if (renderedRevision !== s.revision) {
    chosenText = review && d.reason === 'conflict' ? null : (d.text || null);
    renderedRevision = s.revision;
    if ($('alternatives')) {
      $('alternatives').replaceChildren();
      const seen = new Set();
      const alternatives = s.alternatives.filter(candidate => {
        const key = candidate.trim().replace(/[.!?]+$/, '').toLowerCase();
        if (seen.has(key)) return false;
        seen.add(key); return true;
      });
      if (review && (d.reason === 'conflict' || alternatives.length > 1)) alternatives.forEach(candidate => {
        const button = document.createElement('button'); button.className = 'alternative'; button.textContent = candidate;
        button.onclick = () => { if (!speaking) {chosenText = candidate; updateMessage();} };
        button.dataset.candidate = candidate; $('alternatives').append(button);
      });
    }
  }
  if (!review) { chosenText = d.text || null; if ($('alternatives')) $('alternatives').replaceChildren(); }
  updateMessage();
  text('last-spoken', s.last_spoken ? 'Last spoken: ' + s.last_spoken : '');
  text('result-timing', !s.rehearsal && s.metrics.release_to_text_ms !== undefined ? 'Message ready in ' + (s.metrics.release_to_text_ms / 1000).toFixed(1) + ' s' : '');
  [['metric-release','release_to_text_ms'],['metric-decode','decode_ms'],['metric-fusion','fusion_ms']].forEach(([id,key]) => text(id, s.metrics[key] === undefined ? '—' : s.metrics[key] + ' ms'));
  if (!initialized && $('labels')) {
    const entries = Object.entries(s.vocabulary), enabled = entries.filter(([,v]) => v.enabled ?? v.verified);
    text('vocabulary-summary', enabled.length ? enabled.length + ' gesture mappings are enabled. Review the message before speaking. Live accuracy is still being checked.' : 'No gesture mappings are enabled yet.');
    entries.forEach(([label, entry]) => {
      const name = entry.display_name || label;
      const active = entry.enabled ?? entry.verified;
      const el = document.createElement('span'); el.textContent = name + ' → ' + entry.phrase + (!active ? ' · disabled' : entry.verified ? '' : ' · confirm first'); el.className = entry.verified ? '' : 'pending'; $('labels').append(el);
      const opt = document.createElement('option'); opt.value = label; opt.textContent = name; $('test-label')?.append(opt);
    });
    initialized = true;
  }
}

function updateMessage() {
  if (!state || !$('message')) return;
  const d = state.decision, review = d.status === 'confirmation_needed';
  const message = $('message'), nextText = chosenText || '', changed = message.textContent !== nextText;
  text('message', nextText);
  const processing = !chosenText && ['tail', 'decoding'].includes(state.phase);
  $('message-dock')?.classList.toggle('has-message', Boolean(chosenText));
  $('message-dock')?.classList.toggle('is-processing', processing);
  $('message-dock')?.setAttribute('aria-busy', String(processing));
  hide('message-progress', !processing);
  if (changed) {
    messageAnimation?.cancel(); messageAnimation = null;
    if (nextText && message.animate && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      messageAnimation = message.animate([{opacity:0.2,transform:'translateY(4px)',filter:'blur(3px)'},{opacity:1,transform:'translateY(0)',filter:'blur(0)'}], {duration:280,easing:'cubic-bezier(.22,1,.36,1)'});
    }
  }
  hide('decision-status', !chosenText && !review);
  hide('reason', !chosenText && !review);
  $('message').classList.toggle('empty', !chosenText);
  text('decision-status', review ? 'Check your message' : d.status === 'accepted' ? 'Ready to speak' : ({idle:'Waiting', recording:'Listening', tail:'Finishing', decoding:'Reading lips…'})[state.phase]);
  $('decision-status').classList.toggle('review', review);
  text('reason', !state.speech_available ? 'Speech is unavailable on this device. Check Settings.' : review && d.reason === 'conflict' ? 'The inputs differ. Choose the message you intended.' : review ? 'If these are your words, confirm and speak.' : d.status === 'accepted' ? 'Speak whenever you are ready.' : 'Nothing is spoken until you choose.');
  document.querySelectorAll('.alternative').forEach(b => {const selected = b.dataset.candidate === chosenText; b.classList.toggle('selected', selected); b.setAttribute('aria-pressed', String(selected)); b.disabled = speaking;});
  text('speak-label', review ? 'Confirm & speak' : 'Speak message');
  $('speak').disabled = !chosenText || !state.speech_available || speaking || !['accepted','confirmation_needed'].includes(d.status);
}

async function speakMessage() {
  if (!state || speaking || !chosenText) return;
  const revision = state.revision, message = chosenText, needsConfirmation = state.decision.status === 'confirmation_needed';
  speaking = true; updateMessage();
  try {
    if (needsConfirmation && !await requestAction('confirm', {text:message, revision})) return;
    if (state.revision !== revision || state.decision.status !== 'accepted' || state.decision.text !== message) {
      text('error', 'The message changed. Please check it again.'); return;
    }
    await requestAction('speak', {revision, text:message});
  } finally { speaking = false; updateMessage(); }
}

async function poll() {
  try {
    const response = await fetch('/api/state'); if (!response.ok) throw new Error();
    if (connectionLost) {text('error', ''); connectionLost = false;}
    const result = await response.json(); if (!pendingActions) render(result);
  } catch (_) { connectionLost = true; text('error', 'Connection lost. Reopen QuietBridge when it is ready.'); if ($('speak')) $('speak').disabled = true; if ($('record')) $('record').disabled = true; }
  setTimeout(poll, 180);
}
async function preview() {
  if (!$('camera')) return;
  if (state && !state.rehearsal && state.camera === 'ready') {
    try {
      const res = await fetch('/api/frame'); if (res.ok) {
        const url = URL.createObjectURL(await res.blob()), img = $('camera'), old = img.dataset.url;
        img.src = url; img.dataset.url = url; img.hidden = false; hide('camera-placeholder', true);
        if (old) URL.revokeObjectURL(old);
      }
    } catch (_) {}
  } else { hide('camera', true); hide('camera-placeholder', false); }
  setTimeout(preview, 100);
}
document.querySelectorAll('[data-mode]').forEach(b => b.onclick = () => requestAction('mode', {mode:b.dataset.mode}));
const recordButton = $('record');
recordButton?.addEventListener('pointerdown', e => {if (recordButton.disabled || keyHeld) return; e.preventDefault(); pointerHeld = true; recordButton.setPointerCapture(e.pointerId); requestAction('start');});
function releasePointer() {if (pointerHeld) {pointerHeld = false; requestAction('stop');}}
['pointerup','pointercancel','lostpointercapture'].forEach(event => recordButton?.addEventListener(event, releasePointer));
document.addEventListener('keydown', e => {
  if (e.code === 'Escape' && state && state.phase !== 'idle') {pointerHeld = keyHeld = false; requestAction('cancel'); return;}
  if (!recordButton || ['INPUT','SELECT','TEXTAREA','VIDEO','AUDIO','SUMMARY','A'].includes(e.target.tagName) || e.target.closest?.('.demo-guide') || (e.target.tagName === 'BUTTON' && e.target !== recordButton)) return;
  if (e.code === 'Space' && !e.repeat && !recordButton.disabled && !pointerHeld) {e.preventDefault(); keyHeld = true; requestAction('start');}
});
document.addEventListener('keyup', e => {if (e.code === 'Space' && keyHeld) {e.preventDefault(); keyHeld = false; requestAction('stop');}});
window.addEventListener('blur', () => {if (keyHeld || pointerHeld) {keyHeld = pointerHeld = false; requestAction('stop');}});
if ($('cancel')) $('cancel').onclick = () => {keyHeld = pointerHeld = false; requestAction('cancel');};
if ($('speak')) $('speak').onclick = speakMessage;
if ($('test-submit')) $('test-submit').onclick = async () => {
  const exampleText = state.mode === 'signs' ? '' : $('test-text').value;
  const label = state.mode === 'lips' ? '' : $('test-label').value;
  if (!exampleText.trim() && !label) {text('error', 'Add example words or choose a gesture first.'); return;}
  if (await requestAction('rehearse', {text:exampleText, label})) window.location.assign('/communicate');
};
poll(); preview();
