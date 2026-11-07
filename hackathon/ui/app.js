const $ = id => document.getElementById(id);
const text = (id, value) => { const el = $(id), next = String(value ?? ''); if (el && el.textContent !== next) el.textContent = next; };
const hide = (id, value) => { if ($(id)) $(id).hidden = value; };
let state = null, token = '', actionChain = Promise.resolve(), pointerHeld = false, keyHeld = false;
let speaking = false, pendingActions = 0, renderedRevision = null, renderedAlternatives = '', renderedVocabulary = '', initialized = false, chosenText = null, connectionLost = false;
let browserCameraStream = null, browserCameraStarting = false, browserCameraStopping = false, browserFrameTimer = null, browserFrameCanvas = document.createElement('canvas');
browserFrameCanvas.width = 960; browserFrameCanvas.height = 720;
let editorDirty = false, editorProfile = null, editVersion = 0;
const retiredInstances = new Set();
const ACTION_TIMEOUT_MS = 15000;
const draftKey = profile => 'quietbridge-unsent-v1:' + (profile || 'default');
function storageRead(key) { try { return window.sessionStorage?.getItem(key) || ''; } catch (_) { return ''; } }
function storageWrite(key, value) { try { if (value) window.sessionStorage?.setItem(key, value); else window.sessionStorage?.removeItem(key); } catch (_) {} }
function persistEditor() {
  if (!$('corrected-text') || !state) return;
  storageWrite(draftKey(editorProfile || state.profile_id), editorDirty ? $('corrected-text').value : '');
}
function editMessage() {
  editVersion++;
  editorDirty = $('corrected-text').value !== (state?.decision?.text || '');
  persistEditor(); updateMessage();
  text('draft-status', editorDirty ? 'Unfinished edit kept in this tab. Choose Use this message before speaking it.' : '');
}
function confirmDiscard(message) { return typeof window.confirm === 'function' && window.confirm(message); }
function hasCurrentMessage() { return Boolean(editorDirty || state?.decision?.text || state?.session_text); }
function focusEditor() { if ($('correction-panel')) $('correction-panel').open = true; $('corrected-text')?.focus(); }
function setGuide(open) {
  hide('first-use-guide', !open);
  $('open-guide')?.setAttribute('aria-expanded', String(open));
  if (!open) { try { window.localStorage?.setItem('quietbridge-guide-seen-v1', 'yes'); } catch (_) {} }
}
function renderCapabilities(s) {
  const enabled = Object.values(s.vocabulary || {}).filter(v => v.enabled ?? v.verified).map(v => v.phrase.replace(/[.!?]+$/, ''));
  const backend = s.rehearsal ? 'Preview only, no recognition' : s.signs === 'unavailable' ? 'Gesture input unavailable' : s.sign_input === 'isolated' ? `${s.sign_model?.language?.toUpperCase() || 'Standard'} isolated signs, experimental` : s.sign_input === 'motion' ? 'Personal gestures, experimental' : s.signs === 'loading' ? 'Gesture model loading' : 'Stock hand poses';
  const concepts = enabled.length ? enabled.slice(0, 6).join(', ') + (enabled.length > 6 ? ` and ${enabled.length - 6} more` : '') : 'no enabled concepts';
  text('active-input', `${s.profile_name || 'Stock configuration'} · ${s.rehearsal ? backend : s.mode === 'lips' ? 'Lips only' : s.mode === 'signs' ? backend : 'Lips + ' + backend}. Typing is always available.`);
  text('capability-note', `${backend}: ${concepts}. ${s.rehearsal ? 'Examples do not test recognition.' : 'Only these configured concepts can contribute text. Not full sign-language translation.'}`);
}
async function uploadBrowserFrames() {
  if (!browserCameraStream || !state?.browser_camera || state.rehearsal) return;
  try {
    const video = $('browser-camera');
    if (video?.videoWidth) {
      browserFrameCanvas.getContext('2d').drawImage(video, 0, 0, 960, 720);
      const blob = await new Promise(resolve => browserFrameCanvas.toBlob(resolve, 'image/jpeg', .7));
      if (blob && browserCameraStream && state?.browser_camera && !state.rehearsal) {
        const response = await fetch('/api/frame', {method:'POST', headers:{'Content-Type':'image/jpeg', 'X-Lipflow-Token':token}, body:blob});
        if (response.status === 401 || response.status === 403) {
          const fresh = await fetch('/api/state');
          if (fresh.ok) {
            const next = await fresh.json();
            if (next.token) render(next);
            else text('error', 'Browser camera upload was rejected and no fresh token was provided. Refresh QuietBridge.');
          } else text('error', 'Browser camera upload was rejected. Refresh QuietBridge before trying again.');
        } else if (!response.ok) text('error', `Browser camera upload failed (${response.status}).`);
      }
    }
  } catch (_) { text('error', 'Camera frames could not reach QuietBridge. Finish message to stop the camera, then retry or type below.'); }
  if (browserCameraStream) browserFrameTimer = setTimeout(uploadBrowserFrames, 40);
}
async function startBrowserCamera() {
  if (browserCameraStarting || browserCameraStopping || browserCameraStream) return;
  if (hasCurrentMessage() && !confirmDiscard('Starting the camera begins a new message and replaces the current words. Continue?')) return;
  browserCameraStarting = true;
  const button = $('browser-camera-button');
  if (button) { button.disabled = true; button.textContent = 'Starting camera…'; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio:false, video:{width:{ideal:960}, height:{ideal:720}, frameRate:{ideal:30}}});
    if (!state?.browser_camera || state.rehearsal) { stream.getTracks().forEach(track => track.stop()); return; }
    browserCameraStream = stream;
    $('browser-camera').srcObject = stream;
    browserCameraStream.getVideoTracks()[0]?.addEventListener('ended', () => { void stopCameraSession(); }, {once:true});
    await $('browser-camera').play();
    if (!await requestAction('camera_session', {active:true})) {
      const message = $('error')?.textContent || 'Could not start recognition.';
      stopBrowserCamera();
      await requestAction('camera_session', {active:false});
      text('error', message); return;
    }
    text('error', ''); uploadBrowserFrames();
  } catch (e) {
    const hadCamera = Boolean(browserCameraStream);
    stopBrowserCamera();
    if (hadCamera) await requestAction('camera_session', {active:false});
    text('error', e.name === 'NotAllowedError' ? 'Camera access was blocked. Allow this site in your browser address bar and try again, or type your message below.' : e.name === 'NotFoundError' ? 'No camera was found. Connect one and try again, or type your message below.' : `Could not start the camera: ${e.message || 'check your browser camera permissions'}. You can type below.`);
    focusEditor();
  } finally {
    browserCameraStarting = false;
    if (button) { button.disabled = false; button.textContent = browserCameraStream ? 'Finish message' : 'Start camera'; }
  }
}
async function stopCameraSession() {
  if (browserCameraStopping) return;
  browserCameraStopping = true;
  keyHeld = pointerHeld = false;
  // Stop capture immediately; do not keep sharing frames during a slow action.
  stopBrowserCamera();
  try { await requestAction('camera_session', {active:false}); }
  finally { browserCameraStopping = false; }
}
function endCameraSessionOnUnload() {
  const hadCamera = Boolean(browserCameraStream);
  stopBrowserCamera();
  if (hadCamera && token) {
    void fetch('/api/action/camera_session', {method:'POST', keepalive:true,
      headers:{'Content-Type':'application/json', 'X-Lipflow-Token':token},
      body:JSON.stringify({active:false})}).catch(() => {});
  }
}
function stopBrowserCamera() {
  if (browserFrameTimer) clearTimeout(browserFrameTimer);
  browserFrameTimer = null;
  if (browserCameraStream) browserCameraStream.getTracks().forEach(track => track.stop());
  browserCameraStream = null;
  const video = $('browser-camera'); if (video) video.srcObject = null;
  const button = $('browser-camera-button'); if (button) button.textContent = 'Start camera';
}

const handsCanvas = $('hands-overlay'), handsDraw = handsCanvas?.getContext('2d'), mouthCanvas = $('mouth-overlay'), mouthDraw = mouthCanvas?.getContext('2d');
const handEdges = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
let latestTracking = null, trackingAt = 0, crop = null;

// One owner for mutating message requests. A timeout never retries an action:
// the server may have committed it, so poll and review before trying again.
function requestAction(name, data = {}) {
  const instance = state?.server_instance;
  pendingActions++; updateMessage();
  actionChain = actionChain.then(async () => {
    if (instance && instance !== state?.server_instance) throw new Error('QuietBridge restarted. Review the current message before trying again.');
    const controller = new AbortController();
    let timer;
    try {
      const timedOut = new Promise((_, reject) => {
        timer = setTimeout(() => {
          controller.abort();
          reject(new Error('The request timed out. Its result is uncertain; check the current message before retrying.'));
        }, ACTION_TIMEOUT_MS);
      });
      const request = (async () => {
        const res = await fetch('/api/action/' + name, {method:'POST', signal:controller.signal,
          headers:{'Content-Type':'application/json', 'X-Lipflow-Token':token}, body:JSON.stringify(data)});
        const result = await res.json();
        if (!res.ok) throw new Error(result.error || 'Please try again.');
        return result;
      })();
      const result = await Promise.race([request, timedOut]);
      if (instance && instance !== state?.server_instance) throw new Error('QuietBridge restarted. Review the current message before trying again.');
      if (!render(result)) throw new Error('The request finished, but a newer result is already available. Review the current message.');
      text('error', ''); return true;
    } finally { clearTimeout(timer); }
  }).catch(e => { text('error', `${e.message || 'Connection unavailable.'} Your edit is kept. Review the current message, or copy your typed text.`); return false; }).finally(() => { pendingActions--; updateMessage(); });
  return actionChain;
}

function render(s) {
  if (retiredInstances.has(s.server_instance)) return false;
  const restarted = state?.server_instance && s.server_instance && state.server_instance !== s.server_instance;
  if (state && !restarted && s.revision < state.revision) return false;
  if (restarted) {
    persistEditor(); stopBrowserCamera();
    retiredInstances.add(state.server_instance);
    renderedRevision = null; chosenText = null;
  }
  state = s; if (s.token) token = s.token;
  if (restarted) text('error', 'QuietBridge restarted. The camera is off. Your unfinished edit is kept; review it before using or speaking a message.');
  renderCapabilities(s);
  hide('rehearsal-banner', !s.rehearsal); hide('preview', !s.rehearsal);
  text('camera-status', s.camera); text('lip-status', s.lips); text('hand-status', s.signs);
  text('last-sign', s.last_sign ? s.last_sign.label + ' · ' + s.last_sign.score.toFixed(2) : '—');
  text('sign-input', s.sign_input || '—');
  const signModel = s.sign_model;
  text('sign-model-status', signModel?.language ? `${signModel.language.toUpperCase()} isolated signs · experimental${signModel.sha256 ? ' · ' + signModel.sha256.slice(0,12) : ''}. Record one sign at a time. Not continuous sentence translation.` : 'Standard sign models are not loaded. The current backend recognizes only its configured gestures.');
  text('profile-name', s.profile_name || 'Stock configuration');
  const profileSelect = $('profile-select');
  if (profileSelect && profilesLoaded && !profileSelect.matches(':focus') && s.profile_id) {
    if (profileSelect.value !== s.profile_id) profileSelect.value = s.profile_id;
  }
  const lipModel = s.lip_model || {};
  const lipPending = s.lips === 'warming' || lipModel.status === 'loading';
  const lipUnavailable = s.lips === 'unavailable' || s.lips === 'disabled';
  text('lip-model', lipPending ? 'Loading…' : lipUnavailable ? s.lips : (lipModel.mode || 'stock') + (lipModel.vsr_sha256 ? ' · ' + lipModel.vsr_sha256.slice(0,12) : ''));
  text('language-model', lipPending ? 'Loading…' : lipUnavailable ? s.lips : lipModel.language_model_sha256 ? 'personal · ' + lipModel.language_model_sha256.slice(0,12) : 'stock');
  text('profile-model-status', lipPending || s.signs === 'loading' ? 'Loading models. Wait before starting the camera.' : s.lips === 'unavailable' || s.signs === 'unavailable' ? 'An input model is unavailable. Check Technical details below.' : s.rehearsal ? 'Preview mode. No recognition model is running.' : 'Model status reflects this running process, not just files saved in Personalize.');
  text('component-errors', [s.profile_warning, s.camera_error, s.lip_error, s.sign_error].filter(Boolean).join('\n'));
  const inputWarnings = [s.camera_error, s.sign_error].filter(Boolean);
  text('input-warning', inputWarnings.length ? inputWarnings.join(' ') + ' You can still type your message. Check input details in Settings before retrying the camera.' : '');
  const modeName = {combined:s.sign_input === 'isolated' ? 'Lips and isolated signs' : 'Lips and gestures', lips:'Lips only', signs:s.sign_input === 'isolated' ? 'Isolated signs only' : 'Gestures only'}[s.mode];
  text('mode-caption', modeName);
  if ($('test-text')) $('test-text').disabled = s.mode === 'signs';
  if ($('test-label')) $('test-label').disabled = s.mode === 'lips';
  text('mode-hint', s.phase === 'idle' ? 'Either input can work on its own.' : 'Finish or cancel your recording before changing inputs.');
  document.querySelectorAll('[data-mode]').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === s.mode);
    b.setAttribute('aria-pressed', String(b.dataset.mode === s.mode)); b.disabled = s.phase !== 'idle';
  });
  const ready = s.camera === 'ready';
  const browserCameraAvailable = s.browser_camera === true && !s.rehearsal;
  if (!browserCameraAvailable && browserCameraStream) stopBrowserCamera();
  const browserCameraButton = $('browser-camera-button');
  if (browserCameraButton) {
    browserCameraButton.hidden = !browserCameraAvailable;
    browserCameraButton.textContent = browserCameraStarting ? 'Starting camera…' : browserCameraStopping ? 'Finishing…' : browserCameraStream ? 'Finish message' : 'Start camera';
    browserCameraButton.classList.toggle('is-active', Boolean(browserCameraStream));
    browserCameraButton.disabled = browserCameraStarting || browserCameraStopping || (!browserCameraStream && s.session_finalizing === true);
  }
  text('preview-status', s.rehearsal ? 'Preview · camera off' : ready || browserCameraStream ? 'Camera on' : browserCameraAvailable ? 'Waiting for browser camera' : s.camera === 'unavailable' ? 'Camera unavailable' : 'Getting ready…');
  text('camera-message', s.rehearsal ? 'Your camera is off in preview.' : browserCameraAvailable && !browserCameraStream ? 'Select Start camera to share this browser camera.' : s.camera === 'unavailable' ? 'Allow camera access in your browser, or type your message below.' : 'Getting your camera ready…');
  const record = $('record');
  const motionInput = ['motion', 'isolated'].includes(s.sign_input);
  const explicitMotionSession = motionInput && s.mode !== 'lips' && s.session_active === true;
  const needsLips = s.mode === 'lips' || s.mode === 'combined';
  if (record) {
    const sessionBusy = s.session_finalizing === true || (!explicitMotionSession && (s.auto_recognition === true || s.session_active === true));
    hide('record', s.rehearsal || (s.mode === 'signs' && !motionInput) || sessionBusy);
    record.disabled = sessionBusy || s.rehearsal || !ready || !['idle','recording'].includes(s.phase)
      || (needsLips && s.lips !== 'ready') || (s.mode === 'signs' && s.signs !== 'ready');
    record.classList.toggle('listening', s.phase === 'recording');
    const idleLabel = s.mode === 'signs' ? 'Start gesture recording' : motionInput ? 'Start lips + gesture recording' : 'Start recording words';
    text('record-label', ({recording:'Finish recording', tail:'Finishing…', decoding:'Reading your words…'})[s.phase] || idleLabel);
    record.setAttribute('aria-pressed', String(s.phase === 'recording'));
  }
  hide('cancel', s.phase === 'idle' || (s.session_active === true && !explicitMotionSession) || s.session_finalizing === true);
  text('record-hint', s.rehearsal ? 'Try an example from Settings.' : s.mode === 'signs' && !motionInput ? 'Show a supported gesture to the camera.' : s.lips === 'warming' && needsLips ? 'Getting the lip reader ready…' : s.lips === 'unavailable' && needsLips ? 'Lip reading needs attention in Settings.' : s.mode === 'combined' ? 'Choose Start recording, mouth your words and gesture, then Finish recording.' : 'Press the recording button or Space once to start, and again to finish.');
  text('entry-hint', s.mode === 'signs' ? motionInput ? 'Start recording, make one gesture, then choose Finish recording.' : 'Show a supported gesture, then speak your message.' : s.mode === 'lips' ? 'Start recording, mouth a short sentence, then finish.' : 'Show a supported gesture, or start a recording to combine input.');
  const gesturesReady = s.signs === 'ready' && Object.values(s.vocabulary).some(v => v.enabled ?? v.verified);
  if (!s.rehearsal && !gesturesReady && s.mode !== 'lips') {
    text('entry-hint', s.mode === 'signs' ? 'Gestures need setup. Choose another input in Settings.' : 'Start a recording to mouth a short sentence.');
    text('record-hint', s.mode === 'signs' ? 'Open Settings to check gesture input.' : 'Gesture input is being set up. You can still use your lips once ready.');
  }
  if (s.rehearsal) {
    text('entry-hint', 'Type a message here, or try a sample in Settings. No camera is used.');
    text('record-hint', 'Preview mode does not test recognition.');
  } else if (explicitMotionSession) {
    text('entry-hint', s.sign_input === 'isolated' ? 'Record one sign and its words. Keep face, upper body and both hands in view. Finish recording, then Finish message to review.' : 'Start recording one gesture and its words together. Finish recording, then Finish message to review.');
    text('record-hint', 'Press the recording button or Space to start and finish. Finish message stops the camera.');
  } else if ((s.session_active === true || s.auto_recognition === true) && !s.rehearsal) {
    text('record-hint', 'Mouth a short phrase. Choose Finish message when you are done.');
    text('entry-hint', 'Build your message at your pace. Nothing is spoken yet.');
  } else if (s.session_finalizing === true) {
    text('entry-hint', 'Camera off. Finishing your message for review.');
    text('record-hint', 'Your words stay here while the last result finishes.');
  } else if (browserCameraAvailable && !browserCameraStream) {
    text('entry-hint', s.decision?.text ? 'Check the words below, edit if needed, then speak.' : 'Start the camera, or type a message. You choose what gets spoken.');
    text('record-hint', 'Keep your face and hands in view. No microphone needed.');
  }
  const d = s.decision, review = d.status === 'confirmation_needed';
  const captureBusy = s.session_active || s.session_finalizing || s.phase !== 'idle';
  const flowStep = captureBusy || !d.text ? 'capture' : s.last_spoken === d.text ? 'speak' : 'review';
  ['capture', 'review', 'speak'].forEach(step => {
    const item = $('step-' + step);
    if (item) { if (step === flowStep) item.setAttribute('aria-current', 'step'); else item.removeAttribute('aria-current'); }
  });
  const revisionChanged = renderedRevision !== s.revision;
  const alternativesKey = JSON.stringify([s.alternatives, s.cleanup?.status, s.cleanup?.suggestion, d.status, s.phase]);
  if (revisionChanged) {
    chosenText = review && d.reason === 'conflict' ? null : (d.text || null);
    renderedRevision = s.revision;
    if ($('corrected-text')) {
      if (editorProfile !== (s.profile_id || 'default')) {
        editorProfile = s.profile_id || 'default';
        const saved = storageRead(draftKey(editorProfile));
        editorDirty = Boolean(saved) && saved !== (d.text || '');
        $('corrected-text').value = saved || d.text || '';
        if (saved && !editorDirty) storageWrite(draftKey(editorProfile), '');
      } else if (!editorDirty) $('corrected-text').value = d.text || '';
      text('draft-status', editorDirty ? 'Your unfinished edit is preserved. Use it explicitly, or start a new message to discard it.' : '');
    }
  }
  if (revisionChanged || renderedAlternatives !== alternativesKey) {
    renderedAlternatives = alternativesKey;
    if ($('alternatives')) {
      $('alternatives').replaceChildren();
      const seen = new Set();
      const alternatives = s.alternatives.filter(candidate => {
        const key = candidate === s.cleanup?.suggestion && s.cleanup?.status === 'ready' ? candidate : candidate.trim().replace(/[.!?]+$/, '').toLowerCase();
        if (seen.has(key)) return false;
        seen.add(key); return true;
      });
      if ((review && (d.reason === 'conflict' || alternatives.length > 1)) || (s.cleanup?.status === 'ready' && alternatives.includes(s.cleanup.suggestion))) alternatives.forEach(candidate => {
        const button = document.createElement('button'); button.className = 'alternative';
        button.textContent = candidate === s.cleanup?.suggestion && s.cleanup?.status === 'ready' ? `AI suggestion: ${candidate}` : candidate;
        button.onclick = () => {
          if (speaking) return;
          if (editorDirty && !confirmDiscard('Use these selected words instead of your unfinished edit? The unfinished edit will be replaced.')) return;
          editVersion++; chosenText = candidate;
          if ($('corrected-text')) $('corrected-text').value = candidate;
          editorDirty = false; persistEditor();
          if ($('learn-correction')) $('learn-correction').checked = false;
          text('draft-status', ''); updateMessage();
        };
        button.dataset.candidate = candidate; $('alternatives').append(button);
      });
    }
  }
  if (chosenText && !s.alternatives.includes(chosenText) && chosenText !== d.text) chosenText = d.reason === 'conflict' ? null : (d.text || null);
  if (!review && s.cleanup?.status !== 'ready' && revisionChanged) chosenText = d.text || null;
  if ($('alternatives') && !$('cleanup-status')) {
    const status = document.createElement('p'); status.id = 'cleanup-status'; status.className = 'hint';
    status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite');
    $('alternatives').before(status);
  }
  text('cleanup-status', s.cleanup?.status === 'pending' ? 'Preparing an optional AI suggestion…' : s.cleanup?.status === 'error' ? 'AI suggestion unavailable. Your original message is still available.' : s.cleanup?.status === 'ready' ? 'Review the AI suggestion before choosing it.' : '');
  updateMessage();
  text('last-spoken', s.last_spoken ? 'Last spoken: ' + s.last_spoken : '');
  text('correction-status', s.last_correction || '');
  text('result-timing', !s.rehearsal && s.metrics?.release_to_text_ms !== undefined ? 'Release → text · ' + (s.metrics.release_to_text_ms / 1000).toFixed(1) + ' s' : '');
  [['metric-release','release_to_text_ms'],['metric-decode','decode_ms'],['metric-fusion','fusion_ms']].forEach(([id,key]) => text(id, s.metrics[key] === undefined ? '—' : s.metrics[key] + ' ms'));
  [['metric-camera-fps','camera_input_fps'],['metric-lip-fps','lip_capture_fps']].forEach(([id,key]) => text(id, s.metrics[key] == null ? '—' : s.metrics[key] + ' fps'));
  text('metric-frames', s.metrics.lip_raw_frames == null ? '—' : s.metrics.lip_raw_frames);
  text('metric-gap', s.metrics.lip_max_frame_gap_ms == null ? '—' : s.metrics.lip_max_frame_gap_ms + ' ms');
  const vocabularyKey = JSON.stringify(s.vocabulary);
  if ((!initialized || renderedVocabulary !== vocabularyKey) && $('labels')) {
    renderedVocabulary = vocabularyKey;
    $('labels').replaceChildren();
    if ($('test-label')) $('test-label').replaceChildren(new Option('No gesture', ''));
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
  return true;
}

function updateMessage() {
  if (!state || !$('message')) return;
  const d = state.decision, review = d.status === 'confirmation_needed';
  const sessionActive = state.session_active === true;
  const sessionFinalizing = state.session_finalizing === true;
  const sessionBusy = sessionActive || sessionFinalizing;
  const sessionText = typeof state.session_text === 'string' ? state.session_text.trim() : '';
  const chunkProcessing = ['tail', 'decoding'].includes(state.phase);
  const cleanupPending = state.cleanup?.source_revision === state.revision && state.cleanup?.status === 'pending';
  const processing = sessionFinalizing || chunkProcessing;
  const draft = typeof state.draft_text === 'string' && state.draft_text.trim()
    ? state.draft_text.trim()
    : typeof state.greedy_preview === 'string' && state.greedy_preview.trim()
      ? state.greedy_preview.trim() : '';
  const liveDraft = sessionBusy && Boolean(sessionText);
  const displayText = sessionBusy
    ? sessionText || draft || (sessionFinalizing ? 'Finishing last words…' : 'Your words will appear here.')
    : chosenText || sessionText || (processing ? draft || 'Reading your words…' : review ? 'Which message did you mean?' : 'Your words will appear here.');
  text('message', displayText);
  $('message').classList.toggle('empty', !sessionText && !chosenText && !processing && !draft);
  // Keep accumulated words visible during later chunks; only an empty pending
  // result needs the reading spinner. Live text is a draft, not a final message.
  $('message').classList.toggle('processing', processing && !sessionText);
  $('message').classList.toggle('draft', liveDraft || (processing && Boolean(draft)));
  $('message').style.textDecoration = liveDraft ? 'underline dotted' : '';
  $('message').style.textUnderlineOffset = liveDraft ? '0.18em' : '';
  hide('draft-label', !(liveDraft || processing && Boolean(draft)));
  if ((sessionBusy || processing) && $('alternatives')) $('alternatives').replaceChildren();
  text('decision-status', editorDirty ? 'Unapplied edit' : cleanupPending ? 'Formatting suggestion…' : sessionFinalizing ? 'Finishing last words…' : sessionActive ? 'Recognizing…' : review ? 'Check your message' : d.status === 'accepted' ? 'Ready to speak' : state.phase === 'tail' ? 'Finishing' : state.phase === 'decoding' ? 'Reading lips…' : state.phase === 'recording' ? 'Recording silent input' : state.lips === 'warming' ? 'Warming up' : 'Waiting');
  $('decision-status').classList.toggle('review', review);
  if ($('corrected-text')) $('corrected-text').disabled = sessionBusy || processing;
  text('reason', editorDirty ? 'Your edit has not replaced the displayed message. Choose Use this message before speaking.' : state.lip_error ? 'Some words could not be read. You can edit or type your message. ' + state.lip_error : sessionActive ? 'Live draft. Choose Finish message when you are done.' : sessionFinalizing ? 'Camera off. Finishing the last words.' : !state.speech_available ? 'On-device speech is unavailable. You can show this message or choose Copy message.' : review && d.reason === 'conflict' ? 'The inputs differ. Choose the message you intended.' : review && d.reason === 'partial_agreement' ? 'Only part of the gesture sequence matches. Review the words and alternatives.' : review ? 'If these are your words, confirm and speak.' : d.status === 'accepted' ? 'Speak whenever you are ready.' : 'Nothing is spoken until you choose.');
  document.querySelectorAll('.alternative').forEach(b => {const selected = b.dataset.candidate === chosenText; b.classList.toggle('selected', selected); b.setAttribute('aria-pressed', String(selected)); b.disabled = speaking;});
  text('speak-label', review || (chosenText && chosenText !== d.text) ? 'Confirm & speak' : 'Speak message');
  $('speak').disabled = sessionBusy || processing || editorDirty || connectionLost || pendingActions > 0 || !chosenText || !state.speech_available || speaking || !['accepted','confirmation_needed'].includes(d.status);
  if ($('new-message')) $('new-message').disabled = sessionBusy || processing || speaking || pendingActions > 0 || !hasCurrentMessage();
  if ($('copy-message')) $('copy-message').disabled = !($('corrected-text')?.value.trim() || chosenText);
  if ($('save-correction')) {
    const edited = $('corrected-text')?.value.trim() || '';
    $('save-correction').disabled = sessionBusy || processing || speaking || connectionLost || pendingActions > 0 || !edited || edited === d.text;
  }
}

async function speakMessage() {
  if (!state || speaking || editorDirty || connectionLost || !chosenText) return;
  const revision = state.revision, message = chosenText, version = editVersion, profile = editorProfile;
  const needsConfirmation = state.decision.status === 'confirmation_needed' || message !== state.decision.text;
  speaking = true; updateMessage();
  try {
    if (needsConfirmation && !await requestAction('confirm', {text:message, revision})) return;
    if (editorDirty || editVersion !== version || editorProfile !== profile || chosenText !== message
        || state.revision !== revision || state.decision.status !== 'accepted' || state.decision.text !== message) {
      text('error', 'The message changed. Please check it again.'); return;
    }
    if (await requestAction('speak', {revision, text:message})) {
      $('step-review')?.removeAttribute('aria-current');
      $('step-speak')?.setAttribute('aria-current', 'step');
    }
  } finally { speaking = false; updateMessage(); }
}

function clearTracking() {
  latestTracking = null; crop = null;
  handsDraw?.clearRect(0, 0, 640, 480); mouthDraw?.clearRect(0, 0, 320, 200);
  text('hands-status', 'Waiting for tracking'); text('mouth-status', 'Face not detected');
}
function paintTracking() {
  if (!handsDraw || !mouthDraw || !latestTracking || Date.now() - trackingAt > 1500) { clearTracking(); return; }
  const hands = latestTracking.hands || [], mouth = latestTracking.mouth || {};
  handsDraw.clearRect(0, 0, 640, 480);
  for (const hand of hands) {
    const points = hand.landmarks || [], color = hand.handedness === 'Left' ? '#4cdbab' : '#ffc071';
    handsDraw.strokeStyle = color; handsDraw.fillStyle = color; handsDraw.lineWidth = 2.5;
    for (const [a,b] of handEdges) if (points[a] && points[b]) { handsDraw.beginPath(); handsDraw.moveTo((1-points[a][0])*640, points[a][1]*480); handsDraw.lineTo((1-points[b][0])*640, points[b][1]*480); handsDraw.stroke(); }
    points.forEach((p,i) => { handsDraw.beginPath(); handsDraw.arc((1-p[0])*640,p[1]*480, i===0?5:3,0,Math.PI*2); handsDraw.fill(); });
    if(points.length){handsDraw.font='14px sans-serif';handsDraw.fillText(`${hand.label || hand.handedness || 'Hand'} · ${(hand.score||0).toFixed(2)}`,Math.max(5,(1-points[0][0])*640),Math.max(16,points[0][1]*480-10));}
  }
  text('hands-status', hands.length ? `Tracking ${hands.length} hand${hands.length===1?'':'s'}` : 'No hand detected');
  const box = mouth.mouth_box, image = $('camera');
  if (!box || box.length !== 4 || !image || image.hidden || !image.naturalWidth) { mouthDraw.clearRect(0,0,320,200); text('mouth-status','Face not detected'); crop=null; return; }
  const [bx,by,bw,bh]=box, aspect=1.6*image.naturalHeight/image.naturalWidth;
  const targetW=Math.min(1,Math.max(.18,bw*2.4)), targetH=Math.min(1,Math.max(targetW/aspect,bh*3.2));
  if(!crop) crop={x:1-(bx+bw/2),y:by+bh/2,w:targetW,h:targetH};
  const centerX=1-(bx+bw/2),centerY=by+bh/2;
  crop.x += (centerX-crop.x)*.15; crop.y += (centerY-crop.y)*.15;
  if(Math.abs(targetW-crop.w)/crop.w>.03) crop.w += (targetW-crop.w)*.08;
  if(Math.abs(targetH-crop.h)/crop.h>.03) crop.h += (targetH-crop.h)*.08;
  crop.w=Math.max(.18,Math.min(1,crop.w)); crop.h=Math.max(crop.w/aspect,Math.min(1,crop.h));
  let left=Math.max(0,Math.min(1-crop.w,crop.x-crop.w/2)), top=Math.max(0,Math.min(1-crop.h,crop.y-crop.h/2));
  mouthDraw.clearRect(0,0,320,200);
  mouthDraw.drawImage(image,left*image.naturalWidth,top*image.naturalHeight,crop.w*image.naturalWidth,crop.h*image.naturalHeight,0,0,320,200);
  const path=(points,close)=>{if(!Array.isArray(points)||points.length<2)return;mouthDraw.beginPath();points.forEach((p,i)=>{const x=((1-p[0])-left)/crop.w*320,y=(p[1]-top)/crop.h*200;i?mouthDraw.lineTo(x,y):mouthDraw.moveTo(x,y)});if(close)mouthDraw.closePath();mouthDraw.stroke();points.forEach(p=>{mouthDraw.beginPath();mouthDraw.arc(((1-p[0])-left)/crop.w*320,(p[1]-top)/crop.h*200,2.2,0,Math.PI*2);mouthDraw.fill()})};
  mouthDraw.strokeStyle='#ff5c9a';mouthDraw.fillStyle='#ff8bb3';mouthDraw.lineWidth=2.4;path(mouth.outer_lips,true);mouthDraw.lineWidth=1.8;path(mouth.inner_lips,true);
  text('mouth-status','Tracking lips · live landmarks');
}
async function trackingPoll() {
  try { const r=await fetch('/api/tracking'); if(!r.ok) throw Error(); latestTracking=await r.json(); trackingAt=Date.now(); paintTracking(); }
  catch (_) { if(Date.now()-trackingAt>1500) clearTracking(); }
  setTimeout(trackingPoll,100);
}
async function poll() {
  try {
    const response = await fetch('/api/state'); if (!response.ok) throw new Error();
    if (connectionLost) {text('error', ''); connectionLost = false;}
    const result = await response.json(); render(result);
  } catch (_) { connectionLost = true; persistEditor(); text('error', 'Connection lost. Reconnecting automatically. Your edit is kept in this tab; you can still type or copy it.'); if ($('speak')) $('speak').disabled = true; if ($('record')) $('record').disabled = true; if ($('save-correction')) $('save-correction').disabled = true; }
  setTimeout(poll, 180);
}
async function preview() {
  if (!$('camera')) return;
  if (state && !state.rehearsal && state.camera === 'ready') {
    try {
      // Use the same-origin JPEG URL: blob URLs are intentionally blocked by CSP.
      // Decode before replacing the displayed frame so failed requests never show
      // a broken-image icon or make the mouth canvas draw an incomplete image.
      const next = new Image();
      next.src = '/api/frame?t=' + Date.now();
      await next.decode();
      const img = $('camera');
      img.src = next.src; img.hidden = false; hide('camera-placeholder', true);
    } catch (_) {}
  } else {
    const hasFrozenFrame = $('camera').naturalWidth > 0 && !$('camera').hidden;
    hide('camera', !hasFrozenFrame); hide('camera-placeholder', hasFrozenFrame);
    if (hasFrozenFrame && !browserCameraStream) text('preview-status', 'Camera off · last frame');
  }
  setTimeout(preview, 100);
}
document.querySelectorAll('[data-mode]').forEach(b => b.onclick = () => {
  if (hasCurrentMessage() && b.dataset.mode !== state?.mode && !confirmDiscard('Changing input clears the current message. Continue?')) return;
  return requestAction('mode', {mode:b.dataset.mode});
});
if ($('browser-camera-button')) $('browser-camera-button').onclick = async () => {
  if (browserCameraStream) { await stopCameraSession(); return; }
  if (state?.browser_camera === true && !state.rehearsal) await startBrowserCamera();
};
window.addEventListener('pagehide', endCameraSessionOnUnload);
window.addEventListener('beforeunload', event => {
  persistEditor();
  if (browserCameraStream || state?.session_active) { event.preventDefault(); event.returnValue = ''; }
});
const recordButton = $('record');
async function toggleRecording() {
  if (!state || !recordButton || recordButton.disabled || pendingActions) return;
  await requestAction(state.phase === 'recording' ? 'stop' : 'start');
}
if (recordButton) recordButton.onclick = toggleRecording;
document.addEventListener('keydown', e => {
  if (e.code === 'Escape' && state && state.phase !== 'idle') {requestAction('cancel'); return;}
  if (e.code === 'Escape' && $('first-use-guide') && !$('first-use-guide').hidden) { setGuide(false); $('open-guide')?.focus(); return; }
  if (!recordButton || ['INPUT','SELECT','TEXTAREA','BUTTON','A','SUMMARY'].includes(e.target.tagName) || e.target.isContentEditable) return;
  if (e.code === 'Space' && !e.repeat && !recordButton.disabled) {e.preventDefault(); toggleRecording();}
});
if ($('cancel')) $('cancel').onclick = () => {keyHeld = pointerHeld = false; requestAction('cancel');};
if ($('speak')) $('speak').onclick = speakMessage;
if ($('corrected-text')) $('corrected-text').addEventListener('input', editMessage);
if ($('save-correction')) $('save-correction').onclick = async () => {
  if (!state || pendingActions || $('save-correction').disabled) return;
  const value = $('corrected-text').value;
  const applied = await requestAction('correct', {revision:state.revision, text:value,
                            learn:$('learn-correction').checked});
  if (applied) {
    if ($('corrected-text').value === value) { editorDirty = false; persistEditor(); text('draft-status', ''); }
    $('learn-correction').checked = false;
    updateMessage(); $('speak')?.focus();
  }
};
if ($('new-message')) $('new-message').onclick = async () => {
  if (!state || $('new-message').disabled) return;
  if (hasCurrentMessage() && !confirmDiscard('Start a new message? The current words and unfinished edit will be cleared.')) return;
  const clearingVersion = editVersion, clearingProfile = editorProfile;
  if (await requestAction('new_message', {revision:state.revision})) {
    if (editVersion === clearingVersion && editorProfile === clearingProfile) {
      editorDirty = false; storageWrite(draftKey(editorProfile), '');
      $('corrected-text').value = ''; text('draft-status', '');
    } else {
      persistEditor(); text('draft-status', 'New message started. Your newer unfinished edit is kept; choose Use this message to apply it.');
    }
    $('learn-correction').checked = false;
    updateMessage(); focusEditor();
  }
};
if ($('copy-message')) $('copy-message').onclick = async () => {
  const value = editorDirty ? $('corrected-text').value : chosenText || $('corrected-text').value;
  try { await navigator.clipboard.writeText(value); text('correction-status', 'Message copied.'); }
  catch (_) { focusEditor(); $('corrected-text')?.select(); text('correction-status', 'Select the text and use your device Copy command.'); }
};
if ($('open-guide')) {
  let seen = false; try { seen = window.localStorage?.getItem('quietbridge-guide-seen-v1') === 'yes'; } catch (_) {}
  setGuide(!seen);
  $('open-guide').onclick = () => { const open = $('first-use-guide').hidden; setGuide(open); if (open) $('guide-heading')?.focus(); };
  $('guide-dismiss').onclick = () => { setGuide(false); $('open-guide').focus(); };
  $('guide-type').onclick = () => { setGuide(false); focusEditor(); };
}
document.querySelectorAll('a[href]').forEach(link => link.addEventListener('click', event => {
  persistEditor();
  if ((browserCameraStream || state?.session_active) && !confirmDiscard('Leave this page? The camera will stop and the current recording will finish for review.')) event.preventDefault();
}));
if ($('test-submit')) $('test-submit').onclick = async () => {
  const exampleText = state.mode === 'signs' ? '' : $('test-text').value;
  const label = state.mode === 'lips' ? '' : $('test-label').value;
  if (!exampleText.trim() && !label) {text('error', 'Add example words or choose a gesture first.'); return;}
  if (await requestAction('rehearse', {text:exampleText, label})) window.location.assign('/');
};
let profilesLoaded = false;
async function loadProfilesList() {
  const select = $('profile-select');
  if (!select) return;
  try {
    const res = await fetch('/api/profiles');
    if (!res.ok) throw new Error('Profile request failed');
    const data = await res.json();
    select.replaceChildren();
    (data.profiles || []).forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = `${p.name} (${p.id})`;
      if (p.id === data.active_profile) opt.selected = true;
      select.appendChild(opt);
    });
    profilesLoaded = true;
  } catch (_) { text('profile-switch-error', 'Profiles could not be loaded. Return to your message to keep typing, then retry Settings when connected.'); }
}

async function handleProfileSwitch() {
  const select = $('profile-select');
  const btn = $('profile-switch-btn');
  if (!select || !btn) return;
  const targetId = select.value;
  if (!targetId) return;
  if (browserCameraStream) {
    text('profile-switch-error', 'Stop the browser camera before switching profile.');
    return;
  }
  if (hasCurrentMessage() && !confirmDiscard('Changing or reloading profiles clears the current message. Your unfinished typed edit stays in this tab. Continue?')) return;
  persistEditor();
  text('profile-switch-error', '');
  text('profile-switch-status', 'Switching profile…');
  btn.disabled = true;
  try {
    const res = await fetch('/api/profile/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Lipflow-Token': token },
      body: JSON.stringify({ id: targetId })
    });
    const result = await res.json();
    if (!res.ok) {
      throw new Error(result.error || `Switch failed (${res.status})`);
    }
    text('profile-switch-status', `Selected ${result.profile?.name || targetId}. ${result.lips === 'warming' || result.signs === 'loading' ? 'Models are loading.' : result.lips === 'unavailable' || result.signs === 'unavailable' ? 'Check model errors below.' : 'Model selection refreshed.'}`);
    render(result);
    if ($('labels')) {
      $('labels').replaceChildren();
      if ($('test-label')) $('test-label').replaceChildren(new Option('No gesture', ''));
      initialized = false;
      render(result);
    }
  } catch (err) {
    text('profile-switch-error', err.message);
    text('profile-switch-status', '');
  } finally {
    btn.disabled = false;
  }
}

if ($('profile-reload-btn')) $('profile-reload-btn').onclick = async () => {
  if (!state || !$('profile-select')) return;
  $('profile-select').value = state.profile_id;
  $('profile-reload-btn').disabled = true;
  try { await handleProfileSwitch(); }
  finally { $('profile-reload-btn').disabled = false; }
};
if ($('profile-switch-btn')) $('profile-switch-btn').onclick = handleProfileSwitch;
if ($('profile-select')) loadProfilesList();

poll(); preview(); trackingPoll();
