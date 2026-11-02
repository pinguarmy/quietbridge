const $ = id => document.getElementById(id);
const text = (id, value) => { const el = $(id), next = String(value ?? ''); if (el && el.textContent !== next) el.textContent = next; };
const hide = (id, value) => { if ($(id)) $(id).hidden = value; };
let state = null, token = '', actionChain = Promise.resolve(), pointerHeld = false, keyHeld = false;
let speaking = false, pendingActions = 0, renderedRevision = null, renderedAlternatives = '', initialized = false, chosenText = null, connectionLost = false;
let browserCameraStream = null, browserCameraStarting = false, browserFrameTimer = null, browserFrameCanvas = document.createElement('canvas');
browserFrameCanvas.width = 960; browserFrameCanvas.height = 720;
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
            if (next.token) { token = next.token; render(next); }
            else text('error', 'Browser camera upload was rejected and no fresh token was provided. Refresh QuietBridge.');
          } else text('error', 'Browser camera upload was rejected. Refresh QuietBridge before trying again.');
        } else if (!response.ok) text('error', `Browser camera upload failed (${response.status}).`);
      }
    }
  } catch (_) { /* Keep the camera active; the status endpoint reports server-side errors. */ }
  if (browserCameraStream) browserFrameTimer = setTimeout(uploadBrowserFrames, 40);
}
async function startBrowserCamera() {
  if (browserCameraStarting || browserCameraStream) return;
  browserCameraStarting = true;
  const button = $('browser-camera-button');
  if (button) { button.disabled = true; button.textContent = 'Starting camera…'; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio:false, video:{width:{ideal:960}, height:{ideal:720}, frameRate:{ideal:30}}});
    if (!state?.browser_camera || state.rehearsal) { stream.getTracks().forEach(track => track.stop()); return; }
    browserCameraStream = stream;
    $('browser-camera').srcObject = stream;
    browserCameraStream.getVideoTracks()[0]?.addEventListener('ended', stopBrowserCamera, {once:true});
    text('error', ''); uploadBrowserFrames();
  } catch (e) {
    text('error', e.name === 'NotAllowedError' ? 'Camera access was blocked. Allow camera access for this site in your browser, then try again.' : e.name === 'NotFoundError' ? 'No camera was found. Connect a camera and try again.' : `Could not start the browser camera: ${e.message || 'check browser camera permissions and try again.'}`);
  } finally {
    browserCameraStarting = false;
    if (button) { button.disabled = false; button.textContent = browserCameraStream ? 'Stop camera' : 'Start camera'; }
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
  text('sign-input', s.sign_input || '—');
  text('profile-name', s.profile_name || 'Stock configuration');
  const lipModel = s.lip_model || {};
  text('lip-model', (lipModel.mode || 'stock') + (lipModel.vsr_sha256 ? ' · ' + lipModel.vsr_sha256.slice(0,12) : ''));
  text('language-model', lipModel.language_model_sha256 ? 'personal · ' + lipModel.language_model_sha256.slice(0,12) : 'stock');
  text('component-errors', [s.profile_warning, s.camera_error, s.lip_error, s.sign_error].filter(Boolean).join('\n'));
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
  const browserCameraAvailable = s.browser_camera === true && !s.rehearsal;
  if (!browserCameraAvailable && browserCameraStream) stopBrowserCamera();
  const browserCameraButton = $('browser-camera-button');
  if (browserCameraButton) {
    browserCameraButton.hidden = !browserCameraAvailable;
    browserCameraButton.textContent = browserCameraStarting ? 'Starting camera…' : browserCameraStream ? 'Stop camera' : 'Start camera';
    browserCameraButton.disabled = browserCameraStarting;
  }
  text('preview-status', s.rehearsal ? 'Preview · camera off' : ready || browserCameraStream ? 'Camera on' : browserCameraAvailable ? 'Waiting for browser camera' : s.camera === 'unavailable' ? 'Camera unavailable' : 'Getting ready…');
  text('camera-message', s.rehearsal ? 'Your camera is off in preview.' : browserCameraAvailable && !browserCameraStream ? 'Select Start camera to share this browser camera.' : s.camera === 'unavailable' ? 'Check camera access in Settings.' : 'Getting your camera ready…');
  const record = $('record');
  const motionInput = s.sign_input === 'motion';
  const needsLips = s.mode === 'lips' || s.mode === 'combined';
  if (record) {
    hide('record', s.mode === 'signs' && !motionInput);
    record.disabled = s.rehearsal || !ready || !['idle','recording'].includes(s.phase)
      || (needsLips && s.lips !== 'ready') || (s.mode === 'signs' && s.signs !== 'ready');
    record.classList.toggle('listening', s.phase === 'recording');
    const idleLabel = s.mode === 'signs' ? 'Hold to record a custom gesture' : 'Hold to mouth words';
    text('record-label', ({recording:s.mode === 'signs' ? 'Make one gesture… release to finish' : 'Mouth your words… release to finish', tail:'Finishing…', decoding:'Reading your lips…'})[s.phase] || idleLabel);
  }
  hide('cancel', s.phase === 'idle');
  text('record-hint', s.rehearsal ? 'Try an example from Settings.' : s.mode === 'signs' && !motionInput ? 'Show a supported gesture to the camera.' : s.lips === 'warming' && needsLips ? 'Getting the lip reader ready…' : s.lips === 'unavailable' && needsLips ? 'Lip reading needs attention in Settings.' : s.mode === 'combined' ? 'Hold Space or the button before mouthing and gesturing.' : 'Or hold Space, then release when you finish.');
  text('entry-hint', s.mode === 'signs' ? motionInput ? 'Hold the button, make one custom gesture, then release.' : 'Show a supported gesture, then speak your message.' : s.mode === 'lips' ? 'Hold the button, mouth a sentence, then release.' : 'Show a supported gesture, or hold the button to combine input.');
  const gesturesReady = s.signs === 'ready' && Object.values(s.vocabulary).some(v => v.enabled ?? v.verified);
  if (!s.rehearsal && !gesturesReady && s.mode !== 'lips') {
    text('entry-hint', s.mode === 'signs' ? 'Gestures need setup. Choose another input in Settings.' : 'Hold the button to mouth a short sentence.');
    text('record-hint', s.mode === 'signs' ? 'Open Settings to check gesture input.' : 'Gesture input is being set up. You can still use your lips once ready.');
  }
  const d = s.decision, review = d.status === 'confirmation_needed';
  const revisionChanged = renderedRevision !== s.revision;
  const alternativesKey = JSON.stringify([s.alternatives, s.cleanup?.status, s.cleanup?.suggestion, d.status, s.phase]);
  if (revisionChanged) {
    chosenText = review && d.reason === 'conflict' ? null : (d.text || null);
    renderedRevision = s.revision;
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
        button.onclick = () => { if (!speaking) {chosenText = candidate; updateMessage();} };
        button.dataset.candidate = candidate; $('alternatives').append(button);
      });
    }
  }
  if (!review && s.cleanup?.status !== 'ready' && revisionChanged) chosenText = d.text || null;
  if ($('alternatives') && !$('cleanup-status')) {
    const status = document.createElement('p'); status.id = 'cleanup-status'; status.className = 'hint';
    status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite');
    $('alternatives').before(status);
  }
  text('cleanup-status', s.cleanup?.status === 'pending' ? 'Preparing an optional AI suggestion…' : s.cleanup?.status === 'error' ? 'AI suggestion unavailable. Your original message is still available.' : s.cleanup?.status === 'ready' ? 'Review the AI suggestion before choosing it.' : '');
  updateMessage();
  text('last-spoken', s.last_spoken ? 'Last spoken: ' + s.last_spoken : '');
  text('result-timing', !s.rehearsal && s.metrics?.release_to_text_ms !== undefined ? 'Release → text · ' + (s.metrics.release_to_text_ms / 1000).toFixed(1) + ' s' : '');
  [['metric-release','release_to_text_ms'],['metric-decode','decode_ms'],['metric-fusion','fusion_ms']].forEach(([id,key]) => text(id, s.metrics[key] === undefined ? '—' : s.metrics[key] + ' ms'));
  [['metric-camera-fps','camera_input_fps'],['metric-lip-fps','lip_capture_fps']].forEach(([id,key]) => text(id, s.metrics[key] == null ? '—' : s.metrics[key] + ' fps'));
  text('metric-frames', s.metrics.lip_raw_frames == null ? '—' : s.metrics.lip_raw_frames);
  text('metric-gap', s.metrics.lip_max_frame_gap_ms == null ? '—' : s.metrics.lip_max_frame_gap_ms + ' ms');
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
  const processing = ['tail', 'decoding'].includes(state.phase);
  const draft = typeof state.draft_text === 'string' && state.draft_text.trim()
    ? state.draft_text.trim()
    : typeof state.greedy_preview === 'string' && state.greedy_preview.trim()
      ? state.greedy_preview.trim() : '';
  const displayText = processing ? (draft || 'Reading your words…') : chosenText || (review ? 'Which message did you mean?' : 'Your words will appear here.');
  text('message', displayText);
  $('message').classList.toggle('empty', !chosenText && !processing);
  $('message').classList.toggle('processing', processing);
  $('message').classList.toggle('draft', processing && Boolean(draft));
  hide('draft-label', !(processing && Boolean(draft)));
  if (processing && $('alternatives')) $('alternatives').replaceChildren();
  text('decision-status', review ? 'Check your message' : d.status === 'accepted' ? 'Ready to speak' : state.phase === 'tail' ? 'Finishing' : state.phase === 'decoding' ? 'Reading lips…' : state.phase === 'recording' ? 'Listening' : state.lips === 'warming' ? 'Warming up' : 'Waiting');
  $('decision-status').classList.toggle('review', review);
  text('reason', !state.speech_available ? 'Speech is unavailable on this device. Check Settings.' : review && d.reason === 'conflict' ? 'The inputs differ. Choose the message you intended.' : review && d.reason === 'partial_agreement' ? 'Only part of the gesture sequence matches. Review the words and alternatives.' : review ? 'If these are your words, confirm and speak.' : d.status === 'accepted' ? 'Speak whenever you are ready.' : 'Nothing is spoken until you choose.');
  document.querySelectorAll('.alternative').forEach(b => {const selected = b.dataset.candidate === chosenText; b.classList.toggle('selected', selected); b.setAttribute('aria-pressed', String(selected)); b.disabled = speaking;});
  text('speak-label', review || (chosenText && chosenText !== d.text) ? 'Confirm & speak' : 'Speak message');
  $('speak').disabled = processing || !chosenText || !state.speech_available || speaking || !['accepted','confirmation_needed'].includes(d.status);
}

async function speakMessage() {
  if (!state || speaking || !chosenText) return;
  const revision = state.revision, message = chosenText, needsConfirmation = state.decision.status === 'confirmation_needed' || message !== state.decision.text;
  speaking = true; updateMessage();
  try {
    if (needsConfirmation && !await requestAction('confirm', {text:message, revision})) return;
    if (state.revision !== revision || state.decision.status !== 'accepted' || state.decision.text !== message) {
      text('error', 'The message changed. Please check it again.'); return;
    }
    await requestAction('speak', {revision, text:message});
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
if ($('browser-camera-button')) $('browser-camera-button').onclick = async () => {
  if (browserCameraStream) {
    if (state?.phase !== 'idle') { keyHeld = pointerHeld = false; await requestAction('cancel'); }
    stopBrowserCamera(); return;
  }
  if (state?.browser_camera === true && !state.rehearsal) await startBrowserCamera();
};
window.addEventListener('beforeunload', stopBrowserCamera);
const recordButton = $('record');
recordButton?.addEventListener('pointerdown', e => {if (recordButton.disabled || keyHeld) return; e.preventDefault(); pointerHeld = true; recordButton.setPointerCapture(e.pointerId); requestAction('start');});
function releasePointer() {if (pointerHeld) {pointerHeld = false; requestAction('stop');}}
['pointerup','pointercancel','lostpointercapture'].forEach(event => recordButton?.addEventListener(event, releasePointer));
document.addEventListener('keydown', e => {
  if (e.code === 'Escape' && state && state.phase !== 'idle') {pointerHeld = keyHeld = false; requestAction('cancel'); return;}
  if (!recordButton || ['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName) || (e.target.tagName === 'BUTTON' && e.target !== recordButton)) return;
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
  if (await requestAction('rehearse', {text:exampleText, label})) window.location.assign('/');
};
poll(); preview(); trackingPoll();
