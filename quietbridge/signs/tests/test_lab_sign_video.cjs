const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '..', 'lab.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function helpers() {
  const fields = {
    pairedCaptureKind: {value: 'personal'}, videoConsent: {checked: false},
    pairedGestureSelect: {value: 'WATER'}, pairedPromptSelect: {value: 'I need water'},
  };
  const context = {
    $: id => fields[id], mode: 'paired',
    activeProfile: {id: 'alice', gestures: [{label: 'WATER'}]},
    signModelInfo: {language: 'bsl', manifest_sha256: 'a'.repeat(64), vocabulary: {
      BSL_1019: {phrase: 'water', enabled: true}, BSL_0001: {phrase: 'about', enabled: false},
    }},
  };
  vm.createContext(context);
  const source = script.slice(script.indexOf('function videoRequested()'), script.indexOf('const TARGET_FPS'));
  vm.runInContext(source, context);
  return {context, fields};
}

test('standard-sign capture needs consent and sends the real API flags', () => {
  const {context, fields} = helpers();
  const personal = context.pairedRequest('train', 'personal-session');
  assert.equal(personal.capture_video, false);
  assert.equal(personal.video_consent, false);
  fields.pairedCaptureKind.value = 'video';
  assert.throws(() => context.pairedRequest('test', 'sign-session'), /Agree to save/);
  fields.videoConsent.checked = true;
  fields.pairedGestureSelect.value = 'BSL_1019';
  const video = context.pairedRequest('test', 'sign-session');
  assert.equal(video.capture_video, true);
  assert.equal(video.video_consent, true);
  assert.equal(video.label, 'BSL_1019');
  assert.equal(video.profile, 'alice');
});

test('sign checks use enabled model labels plus negative examples, not personal names', () => {
  const {context, fields} = helpers();
  fields.pairedCaptureKind.value = 'video';
  assert.deepEqual(Array.from(context.pairedItems(), item => item.label), ['BSL_1019', 'REST', 'UNKNOWN']);
  assert.notEqual(context.recordingModeKey('paired'), 'paired');
  assert.match(context.recordingModeKey('paired'), /a{12}/);
  fields.pairedCaptureKind.value = 'personal';
  assert.deepEqual(Array.from(context.pairedItems(), item => item.label), ['WATER']);
  assert.equal(context.recordingModeKey('paired'), 'paired');
});

test('capture-type changes reset consent and cannot alter an active recording', () => {
  const {context, fields} = helpers();
  context.recording = false; context.working = false;
  context.render = () => {};
  fields.videoConsent.checked = true;
  const source = script.slice(script.indexOf("$('pairedCaptureKind').onchange="), script.indexOf("$('pairedGestureSelect').onchange="));
  context.controls = () => {};
  vm.runInContext(source, context);
  fields.pairedCaptureKind.onchange();
  assert.equal(fields.videoConsent.checked, false);
  fields.videoConsent.checked = true;
  context.recording = true;
  fields.pairedCaptureKind.onchange();
  assert.equal(fields.videoConsent.checked, true);
});

test('video sessions remain valid for long profile identifiers', () => {
  const context = {Date, Math};
  vm.createContext(context);
  const fn = script.match(/function getFreshSessionId\(profileId,stageMode,split\)\{[\s\S]*?\n\}/)[0];
  vm.runInContext(fn, context);
  const id = context.getFreshSessionId('a'.repeat(64), 'paired-video-' + 'b'.repeat(12), 'validation');
  assert.match(id, /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/);
});

test('consent controls are unselected on load and validation precedes camera access', () => {
  assert.match(html, /id="videoConsent" type="checkbox">/);
  assert.doesNotMatch(html, /id="videoConsent"[^>]*checked/);
  const record = script.slice(script.indexOf("$('record').onclick=async()=>"), script.indexOf("$('discard').onclick=async()=>"));
  assert.ok(record.indexOf('pairedRequest(currentStage') < record.indexOf('await camera()'));
  assert.match(record, /videoConsent'\)\.checked=false/);
  const discard = script.slice(script.indexOf("$('discard').onclick=async()=>"), script.indexOf('// Model operations'));
  assert.match(discard, /videoConsent'\)\.checked=false/);
});
