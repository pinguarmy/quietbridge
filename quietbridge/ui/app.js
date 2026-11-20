const $ = id => document.getElementById(id);
const text = (id, value) => { const el = $(id), next = String(value ?? ''); if (el && el.textContent !== next) el.textContent = next; };
const hide = (id, value) => { if ($(id)) $(id).hidden = value; };
let state = null, token = '', actionChain = Promise.resolve(), pointerHeld = false, keyHeld = false;
let speaking = false, pendingActions = 0, renderedRevision = null, renderedAlternatives = '', renderedVocabulary = '', initialized = false, chosenText = null, connectionLost = false;
let browserCameraStream = null, browserCameraStarting = false, browserCameraStopping = false, browserCameraGeneration = 0, browserFrameTimer = null, browserFrameCanvas = document.createElement('canvas');
browserFrameCanvas.width = 960; browserFrameCanvas.height = 720;
let editorDirty = false, editorProfile = null, editVersion = 0;
// Text the user explicitly picked from the alternatives for this revision.
let userSelection = null, stopSpeakingPending = false, largeDisplayOpener = null;
// The profile's own quick phrases and this tab's recent messages. Long lists start
// collapsed to the chips that fit in three rows at the current width (12 chips
// where no layout can be measured).
const QUICK_PHRASES_COLLAPSED = 12, CHIP_ROWS_COLLAPSED = 3, RECENT_MAX = 8;
let renderedPhrases = '', showAllPhrases = false, reloadPending = false, renderedRecent = '', showAllRecent = false;
// Quit: once QuietBridge was stopped on purpose, polling continues quietly so a
// restarted app is picked up, but no "connection lost" noise is shown.
let appStopped = false, stoppedInstance = null, quitPending = false;
const retiredInstances = new Set();
const ACTION_TIMEOUT_MS = 15000;
// Upload target stays ~25 fps; the 960x720 JPEG contract is unchanged.
const FRAME_UPLOAD_INTERVAL_MS = 40;
let browserFrameCallback = null, nextFrameUploadAt = 0, lastUploadedVideoTime = null;
// State polling never stops. It is fast while anything can change soon and
// slower only when idle or in a hidden tab. Reconnect checks stay at 1 s or faster.
const POLL_FAST_MS = 180, POLL_IDLE_MS = 1000, POLL_HIDDEN_MS = 3000, POLL_RECONNECT_MS = 1000;
const RECENT_ACTIVITY_MS = 10000;
let pollTimer = null, pollDueAt = 0, pollInFlight = false, lastUserActivity = Date.now();
const draftKey = profile => 'quietbridge-unsent-v1:' + (profile || 'default');
// Recent messages live only in this tab's sessionStorage, per profile; never sent anywhere.
const recentKey = profile => 'quietbridge-recent-v1:' + (profile || 'default');
function storageRead(key) { try { return window.sessionStorage?.getItem(key) ?? null; } catch (_) { return null; } }
function storageWrite(key, value) { try { if (value !== null) window.sessionStorage?.setItem(key, value); else window.sessionStorage?.removeItem(key); } catch (_) {} }
function persistEditor() {
  if (!$('corrected-text') || !state) return;
  storageWrite(draftKey(editorProfile || state.profile_id), editorDirty ? $('corrected-text').value : null);
}
function editMessage() {
  editVersion++;
  editorDirty = $('corrected-text').value !== (state?.decision?.text || '');
  persistEditor(); updateMessage();
  text('draft-status', editorDirty ? 'Unfinished edit kept in this tab.' : '');
}
function confirmDiscard(message) { return typeof window.confirm === 'function' && window.confirm(message); }
function hasCurrentMessage() { return Boolean(editorDirty || state?.decision?.text || state?.session_text); }
function focusEditor() { if ($('correction-panel')) $('correction-panel').open = true; $('corrected-text')?.focus(); }
// The essentials are one line in the intro, so the guide starts closed and
// typing is reachable without scrolling past it.
function setGuide(open) {
  hide('first-use-guide', !open);
  $('open-guide')?.setAttribute('aria-expanded', String(open));
}
function renderCapabilities(s) {
  const enabled = Object.values(s.vocabulary || {}).filter(v => v.enabled ?? v.verified).map(v => v.phrase.replace(/[.!?]+$/, ''));
  const backend = s.rehearsal ? 'Preview only, no recognition' : s.signs === 'unavailable' ? 'Gesture input unavailable' : s.sign_input === 'isolated' ? `${s.sign_model?.language?.toUpperCase() || 'Standard'} isolated signs, experimental` : s.sign_input === 'motion' ? 'Personal gestures, experimental' : s.signs === 'loading' ? 'Gesture model loading' : 'Stock hand poses';
  const concepts = enabled.length ? enabled.slice(0, 6).join(', ') + (enabled.length > 6 ? ` and ${enabled.length - 6} more` : '') : 'no enabled concepts';
  text('active-input', `${s.profile_name || 'Stock configuration'} · ${s.rehearsal ? backend : s.mode === 'lips' ? 'Lips only' : s.mode === 'signs' ? backend : 'Lips + ' + backend}`);
  // The capture hint already says preview mode tests nothing; this line only lists concepts.
  text('capability-note', s.rehearsal ? `Example concepts: ${concepts}.` : `${backend}: ${concepts}. Only these configured concepts can contribute text. Not full sign-language translation.`);
}
// One upload at a time. Uploads follow new video frames (requestVideoFrameCallback
// where available) within a 40 ms budget, so ~25 fps is kept on a 30 fps camera
// and unchanged frames are not re-encoded. Older browsers fall back to a timer.
function scheduleFrameUpload() {
  if (!browserCameraStream) return;
  const generation = browserCameraGeneration, video = $('browser-camera');
  const current = () => browserCameraStream && generation === browserCameraGeneration;
  if (typeof video?.requestVideoFrameCallback === 'function') {
    browserFrameCallback = video.requestVideoFrameCallback(() => {
      browserFrameCallback = null;
      if (!current()) return;
      if (Date.now() < nextFrameUploadAt) scheduleFrameUpload(); else uploadBrowserFrames(true);
    });
  } else {
    browserFrameTimer = setTimeout(() => {
      browserFrameTimer = null;
      if (current()) uploadBrowserFrames(false);
    }, Math.max(10, nextFrameUploadAt - Date.now()));
  }
}
function cancelFrameUpload() {
  if (browserFrameTimer) clearTimeout(browserFrameTimer);
  browserFrameTimer = null;
  const video = $('browser-camera');
  if (browserFrameCallback !== null && typeof video?.cancelVideoFrameCallback === 'function') video.cancelVideoFrameCallback(browserFrameCallback);
  browserFrameCallback = null;
}
function hasNewVideoFrame(video, knownNewFrame) {
  if (knownNewFrame || typeof video.currentTime !== 'number') return true;
  return video.currentTime !== lastUploadedVideoTime;
}
async function uploadBrowserFrames(knownNewFrame = false) {
  if (!browserCameraStream || !state?.browser_camera || state.rehearsal) return;
  const generation = browserCameraGeneration;
  try {
    const video = $('browser-camera');
    if (video?.videoWidth && hasNewVideoFrame(video, knownNewFrame)) {
      const now = Date.now();
      // At most half an interval of catch-up, so a late frame never causes a burst.
      nextFrameUploadAt = Math.max(nextFrameUploadAt, now - FRAME_UPLOAD_INTERVAL_MS / 2) + FRAME_UPLOAD_INTERVAL_MS;
      lastUploadedVideoTime = video.currentTime;
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
  // A stop/start while this upload was in flight owns its own loop.
  if (generation === browserCameraGeneration) scheduleFrameUpload();
}
async function startBrowserCamera() {
  if (browserCameraStarting || browserCameraStopping || browserCameraStream) return;
  if (hasCurrentMessage() && !confirmDiscard('Starting the camera begins a new message and replaces the current words. Continue?')) return;
  browserCameraStarting = true;
  const generation = browserCameraGeneration;
  const button = $('browser-camera-button');
  if (button) { button.disabled = true; text('browser-camera-label', 'Starting camera…'); }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio:false, video:{width:{ideal:960}, height:{ideal:720}, frameRate:{ideal:30}}});
    if (generation !== browserCameraGeneration || !state?.browser_camera || state.rehearsal) { stream.getTracks().forEach(track => track.stop()); return; }
    browserCameraStream = stream;
    $('browser-camera').srcObject = stream;
    browserCameraStream.getVideoTracks()[0]?.addEventListener('ended', () => { void stopCameraSession(); }, {once:true});
    await $('browser-camera').play();
    if (generation !== browserCameraGeneration) return;
    renderPreview();
    if (!await requestAction('camera_session', {active:true})) {
      if (generation !== browserCameraGeneration) return;
      const message = $('error')?.textContent || 'Could not start recognition.';
      stopBrowserCamera();
      await requestAction('camera_session', {active:false});
      text('error', message); return;
    }
    if (generation !== browserCameraGeneration) return;
    text('error', ''); uploadBrowserFrames();
  } catch (e) {
    if (generation !== browserCameraGeneration) return;
    const hadCamera = Boolean(browserCameraStream);
    stopBrowserCamera();
    if (hadCamera) await requestAction('camera_session', {active:false});
    text('error', e.name === 'NotAllowedError' ? 'Camera access was blocked. Allow this site in your browser address bar and try again, or type your message below.' : e.name === 'NotFoundError' ? 'No camera was found. Connect one and try again, or type your message below.' : `Could not start the camera: ${e.message || 'check your browser camera permissions'}. You can type below.`);
    focusEditor();
  } finally {
    browserCameraStarting = false;
    if (button) { button.disabled = false; text('browser-camera-label', browserCameraStream ? 'Finish message' : 'Start camera'); }
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
  cancelFrameUpload();
  browserCameraGeneration++;
  lastUploadedVideoTime = null;
  if (browserCameraStream) browserCameraStream.getTracks().forEach(track => track.stop());
  browserCameraStream = null;
  // The local preview is cleared, not frozen: no picture remains after the camera stops.
  const video = $('browser-camera'); if (video) { video.srcObject = null; video.hidden = true; }
  text('browser-camera-label', 'Start camera');
  renderPreview();
}

const handsCanvas = $('hands-overlay'), handsDraw = handsCanvas?.getContext('2d'), mouthCanvas = $('mouth-overlay'), mouthDraw = mouthCanvas?.getContext('2d');
const handEdges = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
let latestTracking = null, trackingAt = 0, crop = null;

// One owner for mutating message requests. A timeout never retries an action:
// the server may have committed it, so poll and review before trying again.
function requestAction(name, data = {}) {
  const instance = state?.server_instance;
  pendingActions++; updateMessage(); wakePoll();
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
  }).catch(e => {
    if (!instance || instance === state?.server_instance) text('error', `${e.message || 'Connection unavailable.'} Your edit is kept. Review the current message, or copy your typed text.`);
    return false;
  }).finally(() => { pendingActions--; updateMessage(); });
  return actionChain;
}
// Requests that do not change the message (downloads, voice, quit): the same token,
// JSON body and bounded timeout as actions, and never retried automatically.
async function postJson(url, data = {}) {
  const controller = new AbortController();
  let timer;
  try {
    const timedOut = new Promise((_, reject) => {
      timer = setTimeout(() => {
        controller.abort();
        reject(new Error('The request timed out. Its result is uncertain; check the status shown before trying again.'));
      }, ACTION_TIMEOUT_MS);
    });
    const request = (async () => {
      const res = await fetch(url, {method:'POST', signal:controller.signal,
        headers:{'Content-Type':'application/json', 'X-Lipflow-Token':token}, body:JSON.stringify(data)});
      let result = {};
      try { result = await res.json(); } catch (_) {}
      if (!res.ok) throw new Error(result?.error || `QuietBridge refused the request (${res.status}).`);
      return result;
    })();
    return await Promise.race([request, timedOut]);
  } finally { clearTimeout(timer); wakePoll(); }
}
const isStatePayload = value => Boolean(value) && typeof value === 'object' && typeof value.revision === 'number' && Boolean(value.decision);

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
  // Follow the running profile only when it changes. Re-syncing on every poll
  // undid a choice made in the list before the person reached "Use profile",
  // most easily for keyboard users, whose focus has already left the list.
  if (profileSelect && profilesLoaded && s.profile_id && syncedProfileId !== s.profile_id) {
    profileSelect.value = s.profile_id;
    syncedProfileId = s.profile_id;
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
    // The label has its own span so the button's icon survives.
    text('browser-camera-label', browserCameraStarting ? 'Starting camera…' : browserCameraStopping ? 'Finishing…' : browserCameraStream ? 'Finish message' : 'Start camera');
    browserCameraButton.classList.toggle('is-active', Boolean(browserCameraStream));
    browserCameraButton.disabled = browserCameraStarting || browserCameraStopping || (!browserCameraStream && s.session_finalizing === true);
  }
  text('camera-message', cameraMessageText(s));
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
    record.classList.toggle('btn-danger', s.phase === 'recording');
    const idleLabel = s.mode === 'signs' ? 'Start gesture recording' : motionInput ? 'Start lips + gesture recording' : 'Start recording words';
    text('record-label', ({recording:'Finish recording', tail:'Finishing…', decoding:'Reading your words…'})[s.phase] || idleLabel);
    record.setAttribute('aria-pressed', String(s.phase === 'recording'));
  }
  hide('cancel', s.phase === 'idle' || (s.session_active === true && !explicitMotionSession) || s.session_finalizing === true);
  // One capture instruction, inside the camera panel. The intro above the message
  // stays static so changing guidance never moves the message or Speak controls.
  text('record-hint', captureHint(s, {motionInput, explicitMotionSession, needsLips, browserCameraAvailable}));
  renderReloadNotice(s);
  renderSetup(s); renderVoice(s); renderAppInfo(s);
  const d = s.decision, review = d.status === 'confirmation_needed';
  const captureBusy = s.session_active || s.session_finalizing || s.phase !== 'idle';
  const speechError = speechErrorText(s);
  // A failed speech attempt is not shown as spoken.
  const flowStep = captureBusy || !d.text ? 'capture' : s.last_spoken === d.text && !speechError ? 'speak' : 'review';
  ['capture', 'review', 'speak'].forEach(step => {
    const item = $('step-' + step);
    if (item) { if (step === flowStep) item.setAttribute('aria-current', 'step'); else item.removeAttribute('aria-current'); }
  });
  const revisionChanged = renderedRevision !== s.revision;
  // The person's own saved phrases close to what was lip-read: offered as labelled choices, never preselected.
  const phraseMatches = Array.isArray(s.phrase_matches) ? s.phrase_matches.filter(p => typeof p === 'string' && p) : [];
  const alternativesKey = JSON.stringify([s.alternatives, s.cleanup?.status, s.cleanup?.suggestion, d.status, s.phase, phraseMatches]);
  if (revisionChanged) {
    chosenText = review && d.reason === 'conflict' ? null : (d.text || null);
    userSelection = null;
    renderedRevision = s.revision;
    if ($('corrected-text')) {
      if (editorProfile !== (s.profile_id || 'default')) {
        editorProfile = s.profile_id || 'default';
        const saved = storageRead(draftKey(editorProfile));
        editorDirty = saved !== null && saved !== (d.text || '');
        $('corrected-text').value = saved ?? d.text ?? '';
        if (saved !== null && !editorDirty) storageWrite(draftKey(editorProfile), null);
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
      const ownPhraseOffered = alternatives.length > 1 && alternatives.some(candidate => phraseMatches.includes(candidate));
      if ((review && (d.reason === 'conflict' || alternatives.length > 1)) || (s.cleanup?.status === 'ready' && alternatives.includes(s.cleanup.suggestion)) || ownPhraseOffered) alternatives.forEach(candidate => {
        const button = document.createElement('button'); button.type = 'button'; button.className = 'alternative'; button.setAttribute('dir', 'auto');
        button.textContent = candidate === s.cleanup?.suggestion && s.cleanup?.status === 'ready' ? `AI suggestion: ${candidate}` : candidate;
        if (button.textContent === candidate && phraseMatches.includes(candidate)) labelOwnPhrase(button, candidate);
        button.onclick = () => {
          if (speaking) return;
          if (editorDirty && !confirmDiscard('Use these selected words instead of your unfinished edit? The unfinished edit will be replaced.')) return;
          editVersion++; chosenText = candidate; userSelection = candidate;
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
  renderQuickPhrases(s); renderRecent();
  text('speech-error', speechError ? `Speech did not play: ${speechError.replace(/[.!?]*$/, '.')} Use Show large or Copy message instead, or try Speak again.` : '');
  updateMessage();
  renderPreview();
  text('last-spoken', s.last_spoken && !speechError ? 'Last spoken: ' + s.last_spoken : '');
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

// "Your phrase" tag on an alternative; the accessible name reads "Your phrase: <words>".
function labelOwnPhrase(button, candidate) {
  const tag = document.createElement('span'); tag.className = 'alt-tag'; tag.textContent = 'Your phrase';
  const separator = document.createElement('span'); separator.className = 'sr-only'; separator.textContent = ': ';
  const words = document.createElement('span'); words.className = 'alt-text'; words.textContent = candidate;
  button.replaceChildren(tag, separator, words);
  button.classList.add('own-phrase');
}
function captureHint(s, {motionInput, explicitMotionSession, needsLips, browserCameraAvailable}) {
  if (s.rehearsal) return 'Preview mode does not use a camera or test recognition.';
  if (s.session_finalizing === true) return 'Camera off. Your words stay here while the last result finishes.';
  if (explicitMotionSession) return s.sign_input === 'isolated'
    ? 'Record one sign and its words: keep face, upper body and both hands in view. Finish recording, then Finish message to review.'
    : 'Start recording, make one gesture and mouth its words together, then Finish recording. Finish message stops the camera.';
  if (s.session_active === true || s.auto_recognition === true) return 'Mouth a short phrase. Choose Finish message when you are done. Nothing is spoken yet.';
  const lipAsset = setupAsset(s, 'lips');
  if (needsLips && lipAsset && lipAsset.status !== 'ready') return lipAsset.status === 'downloading' ? 'Lip reading is downloading. You can type your message meanwhile.' : 'Lip reading needs its one-time download above. You can type your message meanwhile.';
  if (needsLips && s.setup?.reload_needed === true && s.lips !== 'ready') return 'The download is complete. Choose Start using it above.';
  if (needsLips && s.lips === 'warming') return 'Getting the lip reader ready…';
  if (needsLips && s.lips === 'unavailable') return 'Lip reading needs attention in Settings.';
  const gesturesReady = s.signs === 'ready' && Object.values(s.vocabulary || {}).some(v => v.enabled ?? v.verified);
  if (s.mode !== 'lips' && !gesturesReady) return s.mode === 'signs' ? 'Gestures need setup. Choose another input in Settings.' : 'Gesture input is not ready. You can still use your lips.';
  if (browserCameraAvailable && !browserCameraStream) return 'Start the camera and keep your face and hands in view. No microphone is used.';
  if (s.mode === 'signs' && !motionInput) return 'Show a supported gesture to the camera.';
  return s.mode === 'combined' ? 'Choose Start recording, mouth your words and gesture, then Finish recording.' : 'Press the recording button or Space once to start, and again to finish.';
}
function speechErrorText(s) { return typeof s?.speech_error === 'string' ? s.speech_error.trim() : ''; }

// "A newer model is available": shown from state, never acted on automatically.
// Reloading is the same request as Settings > Reload current models.
function reloadBlockedReason(s) {
  if (browserCameraStream || browserCameraStarting || s.session_active === true || s.session_finalizing === true || s.auto_recognition === true)
    return 'Finish your message and stop the camera first. Models cannot reload while the camera is in use.';
  if (s.phase !== 'idle') return 'Finish or cancel the recording first.';
  if (connectionLost) return 'Waiting for the connection to QuietBridge to return.';
  return '';
}
function renderReloadNotice(s) {
  const needed = s.models_reload_needed === true;
  hide('models-reload-notice', !needed);
  text('models-reload-detail', needed ? (s.models_reload_detail || 'A model saved for this profile differs from the one running now.') : '');
  const blocked = needed ? reloadBlockedReason(s) : '';
  text('models-reload-blocked', blocked);
  if ($('models-reload-button')) $('models-reload-button').disabled = Boolean(blocked) || reloadPending || pendingActions > 0;
  $('profile-reload-btn')?.classList.toggle('reload-needed', needed);
}
async function reloadModels() {
  const button = $('models-reload-button');
  if (!state || !button || button.disabled || state.models_reload_needed !== true) return;
  if (hasCurrentMessage() && !confirmDiscard('Reload models now? This clears the current message. Your unfinished typed edit stays in this tab.')) return;
  persistEditor();
  reloadPending = true; renderReloadNotice(state);
  try {
    if (await requestAction('profile_select', {id: state.profile_id})) {
      text('correction-status', state.lips === 'warming' || state.signs === 'loading' ? 'Models reloaded and loading. The message was cleared.' : 'Models reloaded. The message was cleared.');
    }
  } finally { reloadPending = false; if (state) renderReloadNotice(state); }
}

// ---- First-run setup: one-time model downloads ----
// Shown from state only. Nothing downloads without the person's informed click
// (licence note shown first, explicit consent tick), and nothing reloads by itself.
const SETUP_PURPOSE = {lips: 'read silent lip movements', gestures: 'recognise hand gestures'};
const SETUP_STATUSES = ['ready', 'missing', 'downloading', 'error'];
const setupItems = new Map(), setupPending = new Map(), setupAnnounced = new Map();
let renderedSetupIds = '', setupReloadPending = false;
function setupAssets(s) {
  return Array.isArray(s?.setup?.assets) ? s.setup.assets.filter(a => a && typeof a.id === 'string' && a.id) : [];
}
function setupAsset(s, id) { return setupAssets(s).find(a => a.id === id) || null; }
function setupStatus(a) { return SETUP_STATUSES.includes(a.status) ? a.status : 'missing'; }
function formatBytes(n) {
  const v = Number(n) || 0;
  if (v >= 1e9) return (v / 1e9).toFixed(v >= 1e10 ? 1 : 2) + ' GB';
  if (v >= 1e6) return Math.round(v / 1e6) + ' MB';
  return v >= 1e3 ? Math.round(v / 1e3) + ' kB' : Math.max(0, Math.round(v)) + ' bytes';
}
function setupPercent(a) {
  const done = Number(a.bytes_done) || 0, total = Number(a.bytes_total) || 0;
  return total > 0 ? Math.min(100, Math.floor(done / total * 100)) : null;
}
function setText(el, value) { const next = String(value ?? ''); if (el && el.textContent !== next) el.textContent = next; }
function node(tag, className = '', id = '') {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (id) el.id = id;
  return el;
}
// One item per asset, built once and updated in place so focus and the consent tick survive polling.
function setupItem(asset) {
  if (setupItems.has(asset.id)) return setupItems.get(asset.id);
  const key = asset.id.replace(/[^A-Za-z0-9_-]/g, '_');
  const item = {
    root: node('div', 'setup-item'), head: node('div', 'setup-item-head'), title: node('h3', 'setup-title', `setup-title-${key}`),
    badge: node('span', 'badge setup-badge'), explain: node('p', 'setup-explain', `setup-explain-${key}`),
    failure: node('p', 'callout callout-danger setup-error', `setup-error-${key}`),
    license: node('p', 'setup-license', `setup-license-${key}`),
    consentLabel: node('label', 'check setup-consent'), consent: node('input', '', `setup-consent-${key}`), consentText: node('span'),
    progressWrap: node('div', 'setup-progress'), progress: node('progress', 'progress', `setup-progress-${key}`), bytes: node('p', 'setup-bytes', `setup-bytes-${key}`),
    actions: node('div', 'setup-actions'), download: node('button', 'btn btn-primary btn-s', `setup-download-${key}`),
    cancel: node('button', 'btn btn-s', `setup-cancel-${key}`), hint: node('p', 'hint setup-hint', `setup-hint-${key}`),
  };
  item.failure.setAttribute('role', 'alert');
  item.consent.type = 'checkbox';
  item.consent.setAttribute('aria-describedby', `${item.license.id} ${item.hint.id}`);
  item.consent.addEventListener('change', () => { if (item.consent.checked && item.hint.dataset.reason === 'consent') { item.hint.dataset.reason = ''; setText(item.hint, ''); } });
  item.consentLabel.append(item.consent, item.consentText);
  item.progress.setAttribute('aria-labelledby', item.title.id);
  item.progress.setAttribute('aria-describedby', item.bytes.id);
  item.progressWrap.append(item.progress, item.bytes);
  item.download.type = 'button'; item.cancel.type = 'button';
  item.download.setAttribute('aria-describedby', `${item.explain.id} ${item.hint.id}`);
  item.download.onclick = () => startSetupDownload(asset.id);
  item.cancel.onclick = () => cancelSetupDownload(asset.id);
  item.actions.append(item.download, item.cancel);
  item.head.append(item.title, item.badge);
  item.root.append(item.head, item.explain, item.failure, item.license, item.consentLabel, item.progressWrap, item.actions, item.hint);
  item.root.dataset.asset = asset.id;
  setupItems.set(asset.id, item);
  return item;
}
function updateSetupItem(item, a, s) {
  const status = setupStatus(a), label = a.label || a.id, pending = setupPending.get(a.id);
  const purpose = SETUP_PURPOSE[a.id] || 'use this input';
  const size = typeof a.size_note === 'string' && a.size_note.trim() ? ` (${a.size_note.trim()})` : '';
  const offer = status === 'missing' || status === 'error';
  const otherBusy = s.setup?.busy === true && status !== 'downloading';
  setText(item.title, label);
  setText(item.badge, {ready:'Ready', missing:'Not downloaded', downloading:'Downloading', error:'Stopped'}[status]);
  ['accent', 'warn', 'live'].forEach(t => item.badge.classList.toggle('badge-' + t, t === ({ready:'accent', missing:'warn', downloading:'accent', error:'live'})[status]));
  setText(item.explain, status === 'ready' ? `Downloaded. QuietBridge can ${purpose}.`
    : status === 'downloading' ? 'Downloading now. You can keep typing. Cancel keeps what has arrived, so a later download continues from there.'
    : status === 'error' ? 'The download stopped before it finished. Try again continues from where it stopped.'
    : `To ${purpose}, QuietBridge needs a download${size}.`);
  item.failure.hidden = status !== 'error';
  setText(item.failure, status === 'error' ? (typeof a.error === 'string' && a.error.trim() ? a.error.trim() : 'The download did not finish.') : '');
  const note = typeof a.license_note === 'string' ? a.license_note.trim() : '';
  item.license.hidden = !offer || !note;
  setText(item.license, note ? `Licence: ${note}` : '');
  item.consentLabel.hidden = !offer;
  setText(item.consentText, note ? `I have read the licence note and want to download ${label}${size}.` : `I want to download ${label}${size}.`);
  item.download.hidden = !offer;
  setText(item.download, pending ? 'Starting…' : status === 'error' ? 'Try again' : 'Download');
  item.download.disabled = Boolean(pending) || otherBusy;
  item.cancel.hidden = status !== 'downloading';
  setText(item.cancel, pending ? 'Cancelling…' : 'Cancel download');
  item.cancel.disabled = Boolean(pending);
  item.progressWrap.hidden = status !== 'downloading';
  if (status === 'downloading') {
    const done = Number(a.bytes_done) || 0, total = Number(a.bytes_total) || 0, percent = setupPercent(a);
    if (total > 0) { item.progress.max = total; item.progress.value = Math.min(done, total); }
    else item.progress.removeAttribute('value');
    setText(item.bytes, total > 0 ? `${formatBytes(done)} of ${formatBytes(total)} · ${percent}%` : `${formatBytes(done)} so far`);
  }
  // A message from the person's own action stays until their next action.
  if (!item.hint.dataset.reason) setText(item.hint, otherBusy && offer ? 'Another download is running. Start this one when it finishes.' : '');
}
// Polite, sparse announcements: a change of status and each quarter of progress, never every poll.
function announceSetup(a) {
  const status = setupStatus(a), percent = setupPercent(a), label = a.label || a.id;
  const previous = setupAnnounced.get(a.id), wasDownloading = Boolean(previous?.startsWith('downloading'));
  // Progress of unknown size has no quarters to announce.
  if (status === 'downloading' && percent === null && wasDownloading) return;
  const key = status === 'downloading' ? `downloading:${percent === null ? 0 : Math.floor(percent / 25) * 25}` : status;
  setupAnnounced.set(a.id, key);
  if (previous === undefined || previous === key) return;
  const message = status === 'ready' ? `${label} downloaded.`
    : status === 'downloading' ? (wasDownloading ? `${label} download ${key.split(':')[1]}% done.` : `${label} download started.`)
    : status === 'missing' && wasDownloading ? `${label} download cancelled. What arrived is kept.` : '';
  if (message) text('setup-announce', message);
}
function renderSetup(s) {
  const container = $('setup-assets');
  if (!container) return;
  const hasSetup = Boolean(s.setup) && typeof s.setup === 'object';
  const assets = setupAssets(s), inSettings = Boolean($('downloads'));
  const shown = assets.filter(a => inSettings || setupStatus(a) !== 'ready');
  const reloadNeeded = hasSetup && s.setup.reload_needed === true;
  hide('setup-panel', !hasSetup || (shown.length === 0 && !reloadNeeded));
  hide('downloads', !hasSetup || (assets.length === 0 && !reloadNeeded));
  const offered = shown.some(a => ['missing', 'error'].includes(setupStatus(a)));
  text('setup-intro', offered ? 'Typing works now. The camera needs a one-time download first. Nothing downloads until you choose.'
    : shown.some(a => setupStatus(a) === 'downloading') ? 'Typing works now. The camera is ready to use once the download finishes.' : '');
  const ids = JSON.stringify(shown.map(a => a.id));
  const items = shown.map(a => { const item = setupItem(a); updateSetupItem(item, a, s); announceSetup(a); return item.root; });
  // Re-inserting nodes would drop focus, so the list changes only when its members do.
  if (ids !== renderedSetupIds) { renderedSetupIds = ids; container.replaceChildren(...items); }
  renderSetupReload(s);
}
function renderSetupReload(s) {
  if (!$('setup-reload')) return;
  const needed = s?.setup?.reload_needed === true;
  hide('setup-ready', !needed);
  const blocked = needed ? reloadBlockedReason(s) : '';
  text('setup-reload-blocked', blocked);
  $('setup-reload').disabled = !needed || Boolean(blocked) || setupReloadPending || pendingActions > 0;
}
async function setupRequest(id, url, body, failure) {
  const item = setupItems.get(id);
  setupPending.set(id, true);
  if (item) { item.hint.dataset.reason = ''; setText(item.hint, ''); }
  if (state) renderSetup(state);
  try {
    const result = await postJson(url, body);
    if (isStatePayload(result)) render(result);
    return true;
  } catch (e) {
    // Never retried automatically: the server may already have acted.
    if (item) { item.hint.dataset.reason = 'local'; setText(item.hint, `${failure}: ${e.message || 'QuietBridge did not answer.'}`); }
    return false;
  } finally {
    setupPending.delete(id);
    if (state) renderSetup(state);
  }
}
async function startSetupDownload(id) {
  const item = setupItems.get(id), asset = setupAsset(state, id);
  if (!state || !item || !asset || setupPending.get(id) || !['missing', 'error'].includes(setupStatus(asset)) || item.download.disabled) return;
  if (!item.consent.checked) {
    item.hint.dataset.reason = 'consent';
    setText(item.hint, 'First tick the box to confirm you have read the licence note and want this download.');
    item.consent.focus(); return;
  }
  await setupRequest(id, '/api/setup/download', {id, accept_license: true}, 'The download could not start');
}
async function cancelSetupDownload(id) {
  if (!state || setupPending.get(id) || setupStatus(setupAsset(state, id) || {}) !== 'downloading') return;
  await setupRequest(id, '/api/setup/cancel', {id}, 'The download could not be cancelled');
}
// "Start using it": the existing explicit model reload of the running profile.
async function setupReload() {
  const button = $('setup-reload');
  if (!state || !button || button.disabled || state.setup?.reload_needed !== true) return;
  if (hasCurrentMessage() && !confirmDiscard('Start using the downloaded models now? This clears the current message. Your unfinished typed edit stays in this tab.')) return;
  persistEditor();
  setupReloadPending = true; renderSetupReload(state);
  try {
    if (await requestAction('profile_select', {id: state.profile_id})) {
      const loading = state.lips === 'warming' || state.signs === 'loading';
      const message = loading ? 'Loading the new models. The camera is ready when loading finishes.' : 'The new models are in use.';
      text('correction-status', message); text('setup-status', message);
    }
  } finally { setupReloadPending = false; if (state) renderSetupReload(state); }
}

// ---- Voice (Settings) ----
let voicePending = false, voiceSynced = null, renderedVoices = '';
function renderVoice(s) {
  if (!$('voice')) return;
  const vs = s.speech_settings;
  const valid = Boolean(vs) && typeof vs === 'object';
  hide('voice', !valid);
  if (!valid) return;
  const available = s.speech_available !== false;
  hide('voice-controls', !available);
  text('voice-unavailable', available ? '' : 'On-device speech is not available on this computer, so there is no voice to set. Show large and Copy message still work.');
  for (const rate of ['slow', 'normal', 'fast']) {
    const button = $('voice-rate-' + rate);
    if (button) { button.setAttribute('aria-pressed', String(vs.rate === rate)); button.disabled = voicePending; }
  }
  const voices = (Array.isArray(vs.voices) ? vs.voices : []).filter(v => v && typeof v.id === 'string' && v.id);
  hide('voice-select-field', voices.length === 0);
  text('voice-list-note', voices.length ? '' : 'This computer offers only its standard voice.');
  const select = $('voice-select');
  if (!select) return;
  const listKey = JSON.stringify(voices.map(v => [v.id, v.label]));
  if (listKey !== renderedVoices) {
    renderedVoices = listKey; voiceSynced = null;
    const option = (value, label) => { const o = document.createElement('option'); o.value = value; o.textContent = label; return o; };
    select.replaceChildren(option('', 'System default'), ...voices.map(v => option(v.id, v.label || v.id)));
  }
  // Follow the saved voice only when it changes, so a choice in progress is not undone by polling.
  const saved = typeof vs.voice === 'string' && voices.some(v => v.id === vs.voice) ? vs.voice : '';
  if (!voicePending && voiceSynced !== saved) { select.value = saved; voiceSynced = saved; }
  select.disabled = voicePending;
}
async function saveVoice(change) {
  if (!state || voicePending || !state.speech_settings) return;
  voicePending = true; renderVoice(state);
  text('voice-status', 'Saving…'); text('voice-error', '');
  try {
    const result = await postJson('/api/speech/settings', change);
    if (isStatePayload(result)) render(result);
    text('voice-status', change.rate ? `Saved: ${change.rate} speaking rate. Speak message uses it from now on.` : 'Saved. Speak message uses this voice from now on.');
  } catch (e) {
    text('voice-status', '');
    text('voice-error', `Not saved: ${e.message || 'QuietBridge did not answer.'} The previous setting stays in use.`);
    voiceSynced = null;
  } finally { voicePending = false; if (state) renderVoice(state); }
}

// ---- About line and Quit (Settings) ----
function renderAppInfo(s) {
  const app = s.app && typeof s.app === 'object' ? s.app : null;
  const how = app && ({browser_camera:'using this browser’s camera', server_camera:'using a camera attached to this computer', rehearsal:'in preview mode, without a camera'})[app.started_with];
  text('about-line', app ? [`QuietBridge${typeof app.version === 'string' && app.version ? ' ' + app.version : ''}`, how ? 'started ' + how : ''].filter(Boolean).join(' · ') : '');
  // Only a backend that reports itself can be asked to quit.
  hide('quit-card', !app);
}
async function quitApp() {
  const button = $('quit-button');
  if (!state || !button || quitPending || appStopped) return;
  if (!confirmDiscard('Quit QuietBridge? This stops the camera and QuietBridge on this computer. To use it again, start QuietBridge again.')) return;
  persistEditor();
  quitPending = true; button.disabled = true; text('quit-error', ''); text('quit-status', 'Stopping QuietBridge…');
  try {
    const result = await postJson('/api/shutdown', {});
    if (result?.ok !== true) throw new Error('QuietBridge did not confirm that it stopped.');
    showStoppedScreen();
  } catch (e) {
    text('quit-status', '');
    text('quit-error', `${e.message || 'QuietBridge did not answer.'} It may still be running.`);
  } finally { quitPending = false; button.disabled = false; }
}
function stoppedDraft() {
  if ($('corrected-text') && editorDirty) return $('corrected-text').value;
  return storageRead(draftKey(editorProfile || state?.profile_id)) || '';
}
function showStoppedScreen() {
  persistEditor();
  appStopped = true; stoppedInstance = state?.server_instance || null; connectionLost = false;
  if (browserCameraStream) stopBrowserCamera();
  const draft = stoppedDraft();
  if ($('stopped-draft')) $('stopped-draft').value = draft;
  hide('stopped-draft-block', !draft.trim());
  text('stopped-copy-status', '');
  if ($('main')) $('main').hidden = true;
  hide('stopped-screen', false);
  titleBeforeStop = document.title;
  document.title = 'QuietBridge has stopped';
  $('stopped-heading')?.focus();
  schedulePoll();
}
let titleBeforeStop = '';
function leaveStoppedScreen() {
  appStopped = false; stoppedInstance = null;
  hide('stopped-screen', true);
  if ($('main')) $('main').hidden = false;
  if (titleBeforeStop) document.title = titleBeforeStop;
}

// The person's own phrases. Tapping one applies it exactly like typing it and
// choosing Use this message: it never speaks and never learns.
function phraseChip(phrase, extraClass = '') {
  const button = document.createElement('button');
  button.type = 'button'; button.className = 'quick-phrase chip' + (extraClass ? ' ' + extraClass : '');
  // Visually truncated by CSS; the full text stays the accessible name.
  button.textContent = phrase; button.title = phrase; button.dataset.phrase = phrase;
  button.setAttribute('dir', 'auto');
  button.onclick = () => useQuickPhrase(phrase);
  return button;
}
// Index of the first chip below the collapsed rows, or -1 when everything fits.
// Without measurable layout, falls back to a fixed count.
function chipOverflowIndex(buttons) {
  let rows = 0, rowTop = null;
  for (let i = 0; i < buttons.length; i++) {
    const top = buttons[i].offsetTop;
    if (typeof top !== 'number') return buttons.length > QUICK_PHRASES_COLLAPSED ? QUICK_PHRASES_COLLAPSED : -1;
    if (rowTop === null || top > rowTop + 2) { rowTop = top; if (++rows > CHIP_ROWS_COLLAPSED) return i; }
  }
  return -1;
}
// The same collapse at every width: collapsed lists show what fits in three rows.
// Hidden chips leave the tab order; "Show all" reveals them in place.
function layoutChips(list, more, showAll, noun) {
  if (!list) return;
  const buttons = Array.from(list.children || []);
  buttons.forEach(b => { b.hidden = false; });
  const cut = chipOverflowIndex(buttons);
  if (cut >= 0 && !showAll) buttons.slice(cut).forEach(b => { b.hidden = true; });
  if (more) {
    more.hidden = cut < 0;
    more.textContent = showAll ? `Show fewer ${noun}` : `Show all ${buttons.length} ${noun}`;
    more.setAttribute('aria-expanded', String(showAll));
  }
}
function layoutPhraseLists() {
  layoutChips($('quick-phrases-list'), $('quick-phrases-more'), showAllPhrases, 'phrases');
  layoutChips($('recent-list'), $('recent-more'), showAllRecent, 'recent');
}
function renderQuickPhrases(s) {
  const list = $('quick-phrases-list');
  if (!list) return;
  const phrases = (Array.isArray(s.quick_phrases) ? s.quick_phrases : []).filter(p => typeof p === 'string' && p.trim());
  hide('quick-phrases', phrases.length === 0);
  const key = JSON.stringify(phrases);
  if (key === renderedPhrases) return;
  renderedPhrases = key;
  list.replaceChildren(...phrases.map(phrase => phraseChip(phrase)));
  layoutChips(list, $('quick-phrases-more'), showAllPhrases, 'phrases');
}

// Recent messages in this tab: only text actually dispatched to speech without a
// speech error, or shown with Show large. Choosing one sets the message like a
// phrase; it never speaks. Nothing leaves this tab.
function recentProfile() { return editorProfile || state?.profile_id || 'default'; }
function readRecent(profile = recentProfile()) {
  try {
    const saved = JSON.parse(storageRead(recentKey(profile)) || '[]');
    return Array.isArray(saved) ? saved.filter(m => typeof m === 'string' && m.trim()).slice(0, RECENT_MAX) : [];
  } catch (_) { return []; }
}
function rememberRecent(message, profile = recentProfile()) {
  const value = typeof message === 'string' ? message.trim() : '';
  if (!value) return;
  const list = [value, ...readRecent(profile).filter(m => m !== value)].slice(0, RECENT_MAX);
  storageWrite(recentKey(profile), JSON.stringify(list));
  renderRecent();
}
function renderRecent() {
  const list = $('recent-list');
  if (!list || !state) return;
  const profile = recentProfile(), items = readRecent(profile);
  hide('recent-card', items.length === 0);
  const key = JSON.stringify([profile, items]);
  if (key === renderedRecent) return;
  renderedRecent = key;
  list.replaceChildren(...items.map(message => phraseChip(message, 'recent-chip')));
  layoutChips(list, $('recent-more'), showAllRecent, 'recent');
  if (state) updateMessage();
}
function clearRecent() {
  storageWrite(recentKey(recentProfile()), null);
  showAllRecent = false; renderRecent();
  text('correction-status', 'Recent messages cleared from this tab.');
  focusEditor();
}
function quickPhraseBlocked(v = messageView()) {
  return v.sessionBusy || v.processing || speaking || deviceSpeaking() || connectionLost || pendingActions > 0;
}
function isCurrentMessage(phrase) { return state?.decision?.status === 'accepted' && state.decision.text === phrase; }
async function useQuickPhrase(phrase) {
  if (!state || quickPhraseBlocked() || isCurrentMessage(phrase)) return;
  const editor = $('corrected-text');
  if (editorDirty && editor?.value !== phrase && !confirmDiscard('Use this phrase instead of your unfinished edit? The unfinished edit will be replaced.')) return;
  const version = editVersion, profile = editorProfile;
  if (!await requestAction('correct', {revision:state.revision, text:phrase, learn:false})) return;
  if (editVersion === version && editorProfile === profile) {
    // Replace the edit only if nothing newer was typed while the request ran.
    editorDirty = false;
    if (editor) editor.value = state.decision.text || phrase;
    if ($('learn-correction')) $('learn-correction').checked = false;
    text('draft-status', '');
  } else {
    editorDirty = editor ? editor.value !== (state.decision.text || '') : false;
    text('draft-status', editorDirty ? 'Phrase applied. Your newer unfinished edit is kept; choose Use this message to apply it.' : '');
  }
  persistEditor(); updateMessage();
  // The tapped phrase is now disabled as the current message; keep focus on a delivery control.
  ($('speak') && !$('speak').disabled ? $('speak') : $('show-large'))?.focus();
}

// Snapshot of the message flags shared by the status copy and button guards.
function messageView() {
  const d = state.decision, review = d.status === 'confirmation_needed';
  const sessionActive = state.session_active === true;
  const sessionFinalizing = state.session_finalizing === true;
  const chunkProcessing = ['tail', 'decoding'].includes(state.phase);
  const draft = typeof state.draft_text === 'string' && state.draft_text.trim()
    ? state.draft_text.trim()
    : typeof state.greedy_preview === 'string' && state.greedy_preview.trim()
      ? state.greedy_preview.trim() : '';
  return {
    d, review, sessionActive, sessionFinalizing, draft,
    sessionBusy: sessionActive || sessionFinalizing,
    sessionText: typeof state.session_text === 'string' ? state.session_text.trim() : '',
    processing: sessionFinalizing || chunkProcessing,
    cleanupPending: state.cleanup?.source_revision === state.revision && state.cleanup?.status === 'pending',
  };
}
function deviceSpeaking() { return state?.speaking === true; }
function decisionStatusText(v) {
  if (editorDirty) return 'Unapplied edit';
  if (v.cleanupPending) return 'Formatting suggestion…';
  if (v.sessionFinalizing) return 'Finishing last words…';
  if (v.sessionActive) return 'Recognizing…';
  if (deviceSpeaking()) return 'Speaking…';
  if (v.review) return 'Check your message';
  if (v.d.status === 'accepted') return 'Ready to speak';
  return ({tail:'Finishing', decoding:'Reading lips…', recording:'Recording silent input'})[state.phase]
    || (state.lips === 'warming' ? 'Warming up' : 'Waiting');
}
function reasonText(v) {
  if (editorDirty) return 'Your edit has not replaced the displayed message. Choose Use this message before speaking.';
  if (state.lip_error) return 'Some words could not be read. You can edit or type your message. ' + state.lip_error;
  if (v.sessionActive) return 'Live draft. Choose Finish message when you are done.';
  if (v.sessionFinalizing) return 'Camera off. Finishing the last words.';
  if (!state.speech_available) return 'On-device speech is unavailable. Choose Show large to turn the screen toward someone, or Copy message.';
  if (v.review && v.d.reason === 'conflict') return 'The inputs differ. Choose the message you intended.';
  if (v.review && v.d.reason === 'partial_agreement') return 'Only part of the gesture sequence matches. Review the words and alternatives.';
  if (v.review) return 'If these are your words, confirm and speak.';
  if (v.d.status === 'accepted') return 'Speak or show it whenever you are ready.';
  return 'Nothing is spoken until you choose.';
}
// Show large displays only text the user applied or explicitly selected:
// never an unapplied edit, a live draft or an unchosen conflict.
function largeDisplayText(v) {
  if (editorDirty || v.sessionBusy || v.processing || !chosenText) return null;
  if (v.d.status === 'accepted' && chosenText === v.d.text) return chosenText;
  if (userSelection !== null && chosenText === userSelection) return chosenText;
  return null;
}
function updateMessage() {
  if (!state || !$('message')) return;
  const v = messageView(), {d, review, sessionBusy, sessionText, processing, draft} = v;
  const liveDraft = sessionBusy && Boolean(sessionText);
  const displayText = sessionBusy
    ? sessionText || draft || (v.sessionFinalizing ? 'Finishing last words…' : 'Your words will appear here.')
    : chosenText || sessionText || (processing ? draft || 'Reading your words…' : review ? 'Which message did you mean?' : 'Your words will appear here.');
  text('message', displayText);
  $('message').classList.toggle('empty', !sessionText && !chosenText && !processing && !draft);
  // Long messages use a smaller size and scroll inside the panel, so Speak stays near.
  const longMessage = displayText.length > 160;
  $('message').classList.toggle('long', longMessage);
  if (longMessage) $('message').setAttribute('tabindex', '0'); else $('message').removeAttribute('tabindex');
  // Keep accumulated words visible during later chunks; only an empty pending
  // result needs the reading spinner. Live text is a draft, not a final message.
  $('message').classList.toggle('processing', processing && !sessionText);
  $('message').classList.toggle('draft', liveDraft || (processing && Boolean(draft)));
  $('message').classList.toggle('live-draft', liveDraft);
  // Hold announcements while live words change; the settled result is announced once.
  $('message').setAttribute('aria-busy', String(sessionBusy || processing));
  hide('draft-label', !(liveDraft || processing && Boolean(draft)));
  if ((sessionBusy || processing) && $('alternatives')) $('alternatives').replaceChildren();
  text('decision-status', decisionStatusText(v));
  $('decision-status').classList.toggle('review', review);
  // Badge tone only: an unapplied edit warns, live input and speech are live, a reviewable message is accent.
  const tone = editorDirty ? 'warn' : v.sessionActive || deviceSpeaking() ? 'live' : review || d.status === 'accepted' ? 'accent' : '';
  ['accent', 'warn', 'live'].forEach(t => $('decision-status').classList.toggle('badge-' + t, t === tone));
  if ($('corrected-text')) $('corrected-text').disabled = sessionBusy || processing;
  text('reason', reasonText(v));
  document.querySelectorAll('.alternative').forEach(b => {const selected = b.dataset.candidate === chosenText; b.classList.toggle('selected', selected); b.setAttribute('aria-pressed', String(selected)); b.disabled = speaking;});
  text('speak-label', review || (chosenText && chosenText !== d.text) ? 'Confirm & speak' : 'Speak message');
  $('speak').disabled = sessionBusy || processing || editorDirty || connectionLost || pendingActions > 0 || !chosenText || !state.speech_available || speaking || deviceSpeaking() || !['accepted','confirmation_needed'].includes(d.status);
  // Without on-device speech Speak can never be used; Show large becomes the main action instead.
  hide('speak', !state.speech_available);
  hide('stop-speaking', !deviceSpeaking());
  if ($('stop-speaking')) $('stop-speaking').disabled = stopSpeakingPending;
  if ($('show-large')) $('show-large').disabled = largeDisplayText(v) === null;
  const showLeads = !state.speech_available || Boolean(speechErrorText(state));
  $('show-large')?.classList.toggle('show-large-primary', showLeads);
  // Presentation: exactly one hero action. Show large takes it when speech cannot be relied on;
  // Stop speaking takes it while the device speaks.
  ['btn-primary', 'btn-hero'].forEach(c => { $('show-large')?.classList.toggle(c, showLeads); $('speak')?.classList.toggle(c, !showLeads); });
  $('message-actions')?.classList.toggle('show-leads', showLeads);
  $('message-actions')?.classList.toggle('speaking', deviceSpeaking());
  const phraseBlocked = quickPhraseBlocked(v);
  [...Array.from($('quick-phrases-list')?.children || []), ...Array.from($('recent-list')?.children || [])].forEach(b => {
    const current = isCurrentMessage(b.dataset.phrase);
    b.disabled = phraseBlocked || current;
    b.classList.toggle('current', current);
    if (current) b.setAttribute('aria-current', 'true'); else b.removeAttribute('aria-current');
  });
  if ($('models-reload-button') && state.models_reload_needed === true) $('models-reload-button').disabled = Boolean(reloadBlockedReason(state)) || reloadPending || pendingActions > 0;
  renderSetupReload(state);
  if ($('new-message')) $('new-message').disabled = sessionBusy || processing || speaking || pendingActions > 0 || !hasCurrentMessage();
  if ($('copy-message')) $('copy-message').disabled = !($('corrected-text')?.value.trim() || chosenText);
  if ($('save-correction')) {
    const edited = $('corrected-text')?.value.trim() || '';
    $('save-correction').disabled = sessionBusy || processing || speaking || connectionLost || pendingActions > 0 || !edited || edited === d.text;
    // An unapplied edit makes applying it the obvious next step (Speak is disabled meanwhile).
    $('save-correction').classList.toggle('btn-primary', editorDirty && !$('save-correction').disabled);
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
    if (await requestAction('speak', {revision, text:message}) && !speechErrorText(state)) {
      $('step-review')?.removeAttribute('aria-current');
      $('step-speak')?.setAttribute('aria-current', 'step');
      // Dispatched without a speech error: remembered in this tab only.
      rememberRecent(message, profile);
    }
  } finally { speaking = false; updateMessage(); }
}

function clearTracking() {
  latestTracking = null; crop = null;
  handsDraw?.clearRect(0, 0, 640, 480); mouthDraw?.clearRect(0, 0, 320, 200);
  text('hands-status', 'Waiting for tracking'); text('mouth-status', 'Face not detected');
}
// The picture the mouth crop is drawn from. Both previews appear mirrored: the
// server JPEG is mirrored in its pixels, the local video only through CSS.
function previewSource() {
  const video = $('browser-camera');
  if (browserCameraStream && video && !video.hidden && video.videoWidth) return {el:video, width:video.videoWidth, height:video.videoHeight, mirroredPixels:false};
  const image = $('camera');
  if (image && !image.hidden && image.naturalWidth) return {el:image, width:image.naturalWidth, height:image.naturalHeight, mirroredPixels:true};
  return null;
}
// Light strokes with a dark halo stay legible over any video, in either page theme.
function haloStroke(draw, path) {
  draw.save(); draw.strokeStyle = 'rgba(0,0,0,.6)'; draw.lineWidth += 2.5; path(); draw.stroke(); draw.restore();
  path(); draw.stroke();
}
function haloText(draw, label, x, y) {
  draw.save(); draw.font = '600 15px sans-serif'; draw.lineJoin = 'round'; draw.lineWidth = 4; draw.strokeStyle = 'rgba(0,0,0,.8)';
  draw.strokeText(label, x, y); draw.fillText(label, x, y); draw.restore();
}
function paintHands(hands) {
  handsDraw.clearRect(0, 0, 640, 480);
  for (const hand of hands) {
    const points = hand.landmarks || [], color = hand.handedness === 'Left' ? '#4cdbab' : '#ffc071';
    const at = p => [(1 - p[0]) * 640, p[1] * 480];
    handsDraw.strokeStyle = color; handsDraw.fillStyle = color; handsDraw.lineWidth = 2.5;
    for (const [a, b] of handEdges) if (points[a] && points[b]) haloStroke(handsDraw, () => { handsDraw.beginPath(); handsDraw.moveTo(...at(points[a])); handsDraw.lineTo(...at(points[b])); });
    points.forEach((p, i) => { handsDraw.beginPath(); handsDraw.arc(...at(p), i === 0 ? 5 : 3, 0, Math.PI * 2); handsDraw.fill(); });
    if (points.length) { const [x, y] = at(points[0]); haloText(handsDraw, `${hand.label || hand.handedness || 'Hand'} · ${(hand.score || 0).toFixed(2)}`, Math.max(5, x), Math.max(16, y - 10)); }
  }
  text('hands-status', hands.length ? `Tracking ${hands.length} hand${hands.length === 1 ? '' : 's'}` : 'No hand detected');
}
function paintTracking() {
  if (!handsDraw || !mouthDraw || !latestTracking || Date.now() - trackingAt > 1500) { clearTracking(); return; }
  const hands = latestTracking.hands || [], mouth = latestTracking.mouth || {};
  paintHands(hands);
  const box = mouth.mouth_box, source = previewSource();
  if (!box || box.length !== 4 || !source) { mouthDraw.clearRect(0,0,320,200); text('mouth-status','Face not detected'); crop=null; return; }
  const [bx,by,bw,bh]=box, aspect=1.6*source.height/source.width;
  const targetW=Math.min(1,Math.max(.18,bw*2.4)), targetH=Math.min(1,Math.max(targetW/aspect,bh*3.2));
  if(!crop) crop={x:1-(bx+bw/2),y:by+bh/2,w:targetW,h:targetH};
  const centerX=1-(bx+bw/2),centerY=by+bh/2;
  crop.x += (centerX-crop.x)*.15; crop.y += (centerY-crop.y)*.15;
  if(Math.abs(targetW-crop.w)/crop.w>.03) crop.w += (targetW-crop.w)*.08;
  if(Math.abs(targetH-crop.h)/crop.h>.03) crop.h += (targetH-crop.h)*.08;
  crop.w=Math.max(.18,Math.min(1,crop.w)); crop.h=Math.max(crop.w/aspect,Math.min(1,crop.h));
  let left=Math.max(0,Math.min(1-crop.w,crop.x-crop.w/2)), top=Math.max(0,Math.min(1-crop.h,crop.y-crop.h/2));
  mouthDraw.clearRect(0,0,320,200);
  if (source.mirroredPixels) mouthDraw.drawImage(source.el,left*source.width,top*source.height,crop.w*source.width,crop.h*source.height,0,0,320,200);
  else {
    // Mirrored crop [left, left+w] is source [1-left-w, 1-left], drawn flipped.
    mouthDraw.save(); mouthDraw.translate(320, 0); mouthDraw.scale(-1, 1);
    mouthDraw.drawImage(source.el,(1-left-crop.w)*source.width,top*source.height,crop.w*source.width,crop.h*source.height,0,0,320,200);
    mouthDraw.restore();
  }
  const path=(points,close)=>{if(!Array.isArray(points)||points.length<2)return;mouthDraw.beginPath();points.forEach((p,i)=>{const x=((1-p[0])-left)/crop.w*320,y=(p[1]-top)/crop.h*200;i?mouthDraw.lineTo(x,y):mouthDraw.moveTo(x,y)});if(close)mouthDraw.closePath();mouthDraw.stroke();points.forEach(p=>{mouthDraw.beginPath();mouthDraw.arc(((1-p[0])-left)/crop.w*320,(p[1]-top)/crop.h*200,2.2,0,Math.PI*2);mouthDraw.fill()})};
  mouthDraw.strokeStyle='#ff5c9a';mouthDraw.fillStyle='#ff8bb3';mouthDraw.lineWidth=2.4;path(mouth.outer_lips,true);mouthDraw.lineWidth=1.8;path(mouth.inner_lips,true);
  text('mouth-status','Tracking lips · live landmarks');
}

// Camera loops run only while there is a live picture to show in a visible tab.
let trackingRunning = false, previewRunning = false;
function cameraLive() {
  if (!state || state.rehearsal) return false;
  return state.browser_camera === true ? Boolean(browserCameraStream) : state.camera === 'ready';
}
function trackingWanted() { return Boolean($('hands-overlay')) && cameraLive() && !document.hidden; }
// GET /api/frame is only for a server-owned camera. A browser-owned camera shows its local video.
function serverPreviewWanted() {
  return Boolean($('camera')) && Boolean(state) && !state.rehearsal && state.browser_camera !== true && state.camera === 'ready' && !document.hidden;
}
function ensureCameraLoops() {
  if (!trackingRunning && trackingWanted()) { trackingRunning = true; trackingPoll(); }
  if (!previewRunning && serverPreviewWanted()) { previewRunning = true; preview(); }
}
async function trackingPoll() {
  if (!trackingWanted()) { trackingRunning = false; clearTracking(); return; }
  try { const r=await fetch('/api/tracking'); if(!r.ok) throw Error(); latestTracking=await r.json(); trackingAt=Date.now(); paintTracking(); }
  catch (_) { if(Date.now()-trackingAt>1500) clearTracking(); }
  setTimeout(trackingPoll,100);
}
async function preview() {
  if (!serverPreviewWanted()) { previewRunning = false; renderPreview(); return; }
  try {
    // Use the same-origin JPEG URL: blob URLs are intentionally blocked by CSP.
    // Decode before replacing the displayed frame so failed requests never show
    // a broken-image icon or make the mouth canvas draw an incomplete image.
    const next = new Image();
    next.src = '/api/frame?t=' + Date.now();
    await next.decode();
    const img = $('camera');
    if (serverPreviewWanted()) { img.src = next.src; img.hidden = false; renderPreview(); }
  } catch (_) {}
  setTimeout(preview, 100);
}
function previewStatusText(s) {
  if (s.rehearsal) return 'Preview · camera off';
  if (s.browser_camera === true) return browserCameraStream ? 'Camera on · mirrored' : browserCameraStarting ? 'Starting camera…' : 'Camera off';
  if (s.camera === 'ready') return 'Camera on';
  return s.camera === 'unavailable' ? 'Camera unavailable' : 'Getting ready…';
}
function cameraMessageText(s) {
  if (s.rehearsal) return 'Your camera is off in preview.';
  if (s.browser_camera === true) return 'Select Start camera to share this browser camera.';
  if (s.camera === 'unavailable') return 'Allow camera access in your browser, or type your message below.';
  return 'Getting your camera ready…';
}
// Single owner of which picture is visible: local video, server JPEG or placeholder.
function renderPreview() {
  const image = $('camera'), video = $('browser-camera');
  if (!image && !video) return;
  const serverOwned = Boolean(state) && !state.rehearsal && state.browser_camera !== true;
  if (video) video.hidden = !browserCameraStream;
  if (image && !serverOwned) image.hidden = true;
  if (state) text('preview-status', previewStatusText(state));
  let showing = Boolean(browserCameraStream);
  if (serverOwned && image) {
    showing = !image.hidden && image.naturalWidth > 0;
    // A stopped server camera keeps its last decoded frame, labelled as such.
    if (showing && state.camera !== 'ready') text('preview-status', 'Camera off · last frame');
  }
  hide('camera-placeholder', showing);
  $('camera-preview')?.classList.toggle('live', showing);
  ensureCameraLoops();
}

function pollShouldBeFast() {
  if (!state || pendingActions > 0 || speaking || stopSpeakingPending) return true;
  if (browserCameraStream || browserCameraStarting || browserCameraStopping) return true;
  if (state.phase !== 'idle' || state.session_active === true || state.session_finalizing === true
      || state.auto_recognition === true || state.speaking === true || state.cleanup?.status === 'pending') return true;
  // A server-owned camera can produce gesture results at any moment.
  if (!state.rehearsal && state.browser_camera !== true && state.camera === 'ready') return true;
  return Date.now() - lastUserActivity < RECENT_ACTIVITY_MS;
}
function pollDelay() {
  if (appStopped) return POLL_HIDDEN_MS;
  const actionPending = pendingActions > 0 || speaking || stopSpeakingPending;
  let delay = pollShouldBeFast() ? POLL_FAST_MS : POLL_IDLE_MS;
  if (document.hidden && !actionPending) delay = Math.max(delay, POLL_HIDDEN_MS);
  if (connectionLost) delay = Math.min(delay, POLL_RECONNECT_MS);
  return delay;
}
function schedulePoll(delay = pollDelay()) {
  clearTimeout(pollTimer);
  pollDueAt = Date.now() + delay;
  pollTimer = setTimeout(poll, delay);
}
// Any user action returns polling to the fast rate at once.
function wakePoll() {
  lastUserActivity = Date.now();
  if (!pollInFlight && pollTimer && pollDueAt - Date.now() > POLL_FAST_MS) schedulePoll(0);
}
async function poll() {
  clearTimeout(pollTimer); pollTimer = null; pollInFlight = true;
  try {
    const response = await fetch('/api/state'); if (!response.ok) throw new Error();
    if (appStopped) {
      const next = await response.json();
      // The app that was quit may answer once more while it stops; only a new instance is a restart.
      if (!next?.server_instance || next.server_instance === stoppedInstance) return;
      leaveStoppedScreen(); render(next); retryProfilesList(); return;
    }
    if (connectionLost) {text('error', ''); connectionLost = false;}
    const result = await response.json(); render(result);
    retryProfilesList();
  } catch (_) {
    // Quit on purpose: keep checking quietly for a restart, without reconnect messages.
    if (appStopped) return;
    connectionLost = true; persistEditor();
    text('error', 'Connection lost. Reconnecting automatically. Your edit is kept in this tab; you can still type or copy it.');
    if ($('record')) $('record').disabled = true;
    // Speak, Use this message, phrases and reload all check connectionLost.
    if (state) { updateMessage(); renderReloadNotice(state); renderSetupReload(state); }
  }
  finally { pollInFlight = false; schedulePoll(); }
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
function largeDisplayOpen() { return Boolean($('large-display')) && !$('large-display').hidden; }
// Escape closes the topmost layer first: the large display (a modal), then an
// active recording is cancelled, then the guide closes.
document.addEventListener('keydown', e => {
  wakePoll();
  if (largeDisplayOpen()) {
    if (e.code === 'Escape' || e.key === 'Escape') { e.preventDefault(); closeLargeDisplay(); }
    else if (e.key === 'Tab') { e.preventDefault(); $('large-close')?.focus(); }
    return;
  }
  if (e.code === 'Escape' && state && state.phase !== 'idle') {requestAction('cancel'); return;}
  if (e.code === 'Escape' && $('first-use-guide') && !$('first-use-guide').hidden) { setGuide(false); $('open-guide')?.focus(); return; }
  if (!recordButton || ['INPUT','SELECT','TEXTAREA','BUTTON','A','SUMMARY'].includes(e.target.tagName) || e.target.isContentEditable) return;
  if (e.code === 'Space' && !e.repeat && !recordButton.disabled) {e.preventDefault(); toggleRecording();}
});
document.addEventListener('pointerdown', wakePoll, {passive:true});
document.addEventListener('input', wakePoll, true);
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) wakePoll();
  ensureCameraLoops();
});
if ($('cancel')) $('cancel').onclick = () => {keyHeld = pointerHeld = false; requestAction('cancel');};
if ($('speak')) $('speak').onclick = speakMessage;
if ($('corrected-text')) $('corrected-text').addEventListener('input', editMessage);
async function applyEditedMessage() {
  if (!state || pendingActions || $('save-correction').disabled) return;
  const value = $('corrected-text').value;
  const applied = await requestAction('correct', {revision:state.revision, text:value,
                            learn:$('learn-correction').checked});
  if (applied) {
    if ($('corrected-text').value === value) { editorDirty = false; persistEditor(); text('draft-status', ''); }
    $('learn-correction').checked = false;
    updateMessage(); $('speak')?.focus();
  }
}
if ($('save-correction')) $('save-correction').onclick = applyEditedMessage;
// Ctrl+Enter / Cmd+Enter in the editor = Use this message, with the same guards. It never speaks.
if ($('corrected-text')) $('corrected-text').addEventListener('keydown', e => {
  if (e.key !== 'Enter' || !(e.ctrlKey || e.metaKey) || e.isComposing) return;
  e.preventDefault();
  if ($('save-correction')) return applyEditedMessage();
});
async function stopSpeaking() {
  if (stopSpeakingPending || !token) return;
  // Sent directly, not queued behind a slow action. Idempotent; never retried automatically.
  stopSpeakingPending = true; updateMessage(); wakePoll();
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), ACTION_TIMEOUT_MS);
  try {
    const res = await fetch('/api/action/stop_speaking', {method:'POST', signal:controller.signal,
      headers:{'Content-Type':'application/json', 'X-Lipflow-Token':token}, body:'{}'});
    const result = await res.json();
    if (!res.ok) throw new Error(result.error || 'Speech could not be stopped.');
    render(result);
  } catch (e) {
    text('error', `${e.name === 'AbortError' ? 'Stopping speech timed out.' : e.message || 'Speech could not be stopped.'} Check whether it is still speaking, then try again.`);
  } finally { clearTimeout(timer); stopSpeakingPending = false; updateMessage(); }
}
if ($('stop-speaking')) $('stop-speaking').onclick = stopSpeaking;
// Large-type display for turning the screen toward someone. Shows only text the
// user applied or selected. It never speaks, learns or changes the message.
function largeTextSize(value) {
  const n = value.length;
  return n <= 24 ? 'size-xl' : n <= 70 ? 'size-l' : n <= 180 ? 'size-m' : n <= 500 ? 'size-s' : 'size-xs';
}
function openLargeDisplay() {
  if (!state || !$('large-display')) return;
  const value = largeDisplayText(messageView());
  if (value === null) return;
  largeDisplayOpener = $('show-large');
  const display = $('large-text');
  display.textContent = value;
  ['size-xl', 'size-l', 'size-m', 'size-s', 'size-xs'].forEach(size => display.classList.toggle(size, size === largeTextSize(value)));
  if ($('main')) $('main').inert = true;
  $('large-display').hidden = false;
  document.body?.classList?.add('large-open');
  $('large-close')?.focus();
  rememberRecent(value);
}
function closeLargeDisplay() {
  if (!largeDisplayOpen()) return;
  $('large-display').hidden = true;
  if ($('main')) $('main').inert = false;
  document.body?.classList?.remove('large-open');
  // Return focus to the opener; if it became unavailable meanwhile, to the editor.
  const opener = largeDisplayOpener; largeDisplayOpener = null;
  if (opener && !opener.disabled) opener.focus(); else focusEditor();
}
if ($('show-large')) $('show-large').onclick = openLargeDisplay;
if ($('large-close')) $('large-close').onclick = closeLargeDisplay;
if ($('new-message')) $('new-message').onclick = async () => {
  if (!state || $('new-message').disabled) return;
  if (hasCurrentMessage() && !confirmDiscard('Start a new message? The current words and unfinished edit will be cleared.')) return;
  const clearingVersion = editVersion, clearingProfile = editorProfile;
  if (await requestAction('new_message', {revision:state.revision})) {
    if (editVersion === clearingVersion && editorProfile === clearingProfile) {
      editorDirty = false; storageWrite(draftKey(editorProfile), null);
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
if ($('models-reload-button')) $('models-reload-button').onclick = reloadModels;
if ($('quick-phrases-more')) $('quick-phrases-more').onclick = () => {
  showAllPhrases = !showAllPhrases;
  layoutChips($('quick-phrases-list'), $('quick-phrases-more'), showAllPhrases, 'phrases');
  updateMessage();
};
if ($('recent-more')) $('recent-more').onclick = () => {
  showAllRecent = !showAllRecent;
  layoutChips($('recent-list'), $('recent-more'), showAllRecent, 'recent');
  updateMessage();
};
if ($('recent-clear')) $('recent-clear').onclick = clearRecent;
// Rows depend on the width, so a resize or rotation re-fits the collapsed lists.
let chipLayoutFrame = null;
window.addEventListener('resize', () => {
  if (chipLayoutFrame !== null) return;
  const later = typeof window.requestAnimationFrame === 'function' ? window.requestAnimationFrame.bind(window) : fn => setTimeout(fn, 50);
  chipLayoutFrame = later(() => { chipLayoutFrame = null; layoutPhraseLists(); });
});
if ($('setup-reload')) $('setup-reload').onclick = setupReload;
for (const rate of ['slow', 'normal', 'fast']) if ($('voice-rate-' + rate)) $('voice-rate-' + rate).onclick = () => {
  if (state?.speech_settings?.rate !== rate) return saveVoice({rate});
};
if ($('voice-select')) $('voice-select').addEventListener('change', () => saveVoice({voice: $('voice-select').value}));
if ($('quit-button')) $('quit-button').onclick = quitApp;
if ($('stopped-copy')) $('stopped-copy').onclick = async () => {
  try { await navigator.clipboard.writeText($('stopped-draft').value); text('stopped-copy-status', 'Copied.'); }
  catch (_) { $('stopped-draft').focus(); $('stopped-draft').select(); text('stopped-copy-status', 'Select the text and use your device Copy command.'); }
};
if ($('open-guide')) {
  setGuide(false);
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
let profilesLoaded = false, profilesLoading = false, profilesRetryAt = 0, syncedProfileId = null;
const PROFILES_RETRY_MS = 5000;
function profilePlaceholder(label) {
  const opt = document.createElement('option');
  opt.value = ''; opt.textContent = label; opt.disabled = true; opt.selected = true;
  return opt;
}
// A failed list is retried automatically (a safe GET) once state polling succeeds again.
async function loadProfilesList() {
  const select = $('profile-select');
  if (!select || profilesLoading) return;
  profilesLoading = true;
  try {
    const res = await fetch('/api/profiles');
    if (!res.ok) throw new Error('Profile request failed');
    const data = await res.json();
    const profiles = data.profiles || [];
    select.replaceChildren(...(profiles.length ? [] : [profilePlaceholder('No saved profiles yet')]));
    profiles.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = `${p.name} (${p.id})`;
      if (p.id === data.active_profile) opt.selected = true;
      select.appendChild(opt);
    });
    if ($('profile-switch-btn')) $('profile-switch-btn').disabled = profiles.length === 0;
    text('profile-switch-error', '');
    text('profile-list-status', profiles.length ? '' : 'Create a profile in Personalize to keep your own gestures and phrases.');
    profilesLoaded = true;
    syncedProfileId = data.active_profile || null;
  } catch (_) {
    profilesRetryAt = Date.now() + PROFILES_RETRY_MS;
    select.replaceChildren(profilePlaceholder('Profiles unavailable'));
    if ($('profile-switch-btn')) $('profile-switch-btn').disabled = true;
    text('profile-switch-error', 'Profiles could not be loaded. Trying again automatically. Your message and typing are not affected.');
  } finally { profilesLoading = false; }
}
function retryProfilesList() {
  if ($('profile-select') && !profilesLoaded && !profilesLoading && Date.now() >= profilesRetryAt) loadProfilesList();
}

async function handleProfileSwitch(targetId = $('profile-select')?.value) {
  const btn = $('profile-switch-btn');
  if (!btn || !targetId) return;
  if (browserCameraStream) {
    text('profile-switch-error', 'Stop the browser camera before switching profile.');
    return;
  }
  const blocked = state ? reloadBlockedReason(state) : '';
  if (blocked) { text('profile-switch-error', blocked); return; }
  if (hasCurrentMessage() && !confirmDiscard('Changing or reloading profiles clears the current message. Your unfinished typed edit stays in this tab. Continue?')) return;
  persistEditor();
  text('profile-switch-error', '');
  text('profile-switch-status', 'Switching profile…');
  btn.disabled = true;
  // Bounded like every action: a timeout is reported as uncertain and never retried.
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), ACTION_TIMEOUT_MS);
  try {
    const res = await fetch('/api/profile/select', {
      method: 'POST', signal: controller.signal,
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
    text('profile-switch-error', err.name === 'AbortError' ? 'The request timed out. Its result is uncertain; check which profile is shown above before trying again.' : err.message);
    text('profile-switch-status', '');
  } finally {
    clearTimeout(timer);
    btn.disabled = !profilesLoaded;
  }
}

// Reload always targets the running profile, whatever the list shows or whether it loaded.
if ($('profile-reload-btn')) $('profile-reload-btn').onclick = async () => {
  if (!state?.profile_id) return;
  $('profile-reload-btn').disabled = true;
  try { await handleProfileSwitch(state.profile_id); }
  finally { $('profile-reload-btn').disabled = false; }
};
if ($('profile-switch-btn')) $('profile-switch-btn').onclick = () => handleProfileSwitch();
if ($('profile-select')) loadProfilesList();

poll();
