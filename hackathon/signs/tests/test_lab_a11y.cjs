// Contract tests for lab.html accessibility/resume behaviour against a fake DOM.
// They check script wiring only, not real browser, camera or screen-reader output.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '..', 'lab.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const style = html.match(/<style>([\s\S]*?)<\/style>/)[1];

test('status note does not rewrite identical text (no repeated screen-reader announcements)', () => {
  const source = script.match(/function note\(text,error=false\)\{[\s\S]*?\n\}/)[0];
  let writes = 0;
  const status = {_t: '', className: '', get textContent() { return this._t; }, set textContent(v) { writes++; this._t = v; }};
  const pill = {textContent: '', classList: {toggle() {}}};
  const context = {$: id => (id === 'status' ? status : pill), recording: false};
  vm.createContext(context); vm.runInContext(source, context);
  context.note('Video limit reached.', true);
  context.note('Video limit reached.', true);
  context.note('Video limit reached.', true);
  assert.equal(writes, 1);
  context.note('Saved.');
  assert.equal(writes, 2);
  assert.equal(status.className, '');
});

test('per-frame recording progress goes to a non-live meter, not the status region', () => {
  const send = script.slice(script.indexOf('async function sendFrame()'), script.indexOf('async function camera()'));
  assert.match(send, /\$\('recordingMeter'\)\.textContent=/);
  // Only the limit warning and the error path may call note() from the frame loop.
  assert.equal((send.match(/note\(/g) || []).length, 2);
  assert.match(html, /<p id="recordingMeter" class="meter"><\/p>/);
  assert.doesNotMatch(html, /id="recordingMeter"[^>]*aria-live/);
  assert.match(html, /id="status" role="status" aria-live="polite"/);
});

test('an open lip trial is resumed as lips even though state.recording is also true', () => {
  const start = script.slice(script.indexOf("request('/api/state').then("));
  assert.ok(start.indexOf('s.mouth?.recording') < start.indexOf('}else if(s.recording){'));
  assert.doesNotMatch(start, /mode==='gestures'&&s\.recording/);
});

test('stage and mode buttons expose their state, and modes hide where they do not apply', () => {
  const render = script.slice(script.indexOf('function render(){'), script.indexOf('function checkStageCompletion()'));
  assert.match(render, /setAttribute\('aria-current','step'\)/);
  assert.match(render, /setAttribute\('aria-pressed',String\(on\)\)/);
  assert.match(render, /\$\('modeGroup'\)\.hidden=currentStage==='setup'\|\|currentStage==='models'/);
});

test('profile selector shows names and only adds the id when names collide', () => {
  const source = script.match(/async function loadProfiles\(\)\{[\s\S]*?\n\}/)[0];
  const options = [];
  const sel = {set innerHTML(v) { options.length = 0; }, appendChild(o) { options.push(o); }};
  const context = {
    $: () => sel, allProfiles: [], activeProfile: null,
    document: {createElement: () => ({})},
    request: async () => ({active_profile: 'default', profiles: [
      {id: 'default', name: 'Default Profile'}, {id: 'a1', name: 'Sam'}, {id: 'a2', name: 'Sam'}]}),
  };
  vm.createContext(context); vm.runInContext(source, context);
  return context.loadProfiles().then(() => {
    assert.deepEqual(options.map(o => o.textContent), ['Default Profile', 'Sam (a1)', 'Sam (a2)']);
  });
});

test('model errors keep the server message and add a next step', () => {
  const source = script.match(/function modelErrorHelp\(message\)\{[\s\S]*?\n\}/)[0];
  const context = {};
  vm.createContext(context); vm.runInContext(source, context);
  const help = context.modelErrorHelp('Independent train and validation recordings are required');
  assert.match(help, /^Independent train and validation recordings are required\. Record at least one Practice session/);
  assert.equal(context.modelErrorHelp('Disk full'), 'Disk full');
  assert.match(html, /id="modelError" role="alert"/);
});

test('user-facing copy avoids developer flags and internal ids', () => {
  const visible = html.replace(/<script>[\s\S]*?<\/script>/, '').replace(/<style>[\s\S]*?<\/style>/, '');
  for (const jargon of ['--rehearsal', '--sign-model', 'Prompt text', 'Back to Practice', 'this Mac', 'Multimodal']) {
    assert.ok(!visible.includes(jargon), jargon);
    assert.ok(!script.includes(jargon), jargon);
  }
  // Safety disclosures stay.
  assert.match(visible, /not a BSL or ASL course/);
  assert.match(visible, /at most 240 frames or 10 seconds/);
  assert.match(script, /Nothing is recorded until you start practising/);
});

test('theme uses shared tokens, has dark mode and keeps a 13px type floor', () => {
  for (const token of ['--ink:#182d39', '--muted:#536773', '--accent:#086b68', '--paper:#f3f6f8', '--danger:#a73831']) assert.ok(style.includes(token), token);
  assert.match(style, /@media\(prefers-color-scheme:dark\)\{\s*:root\{[\s\S]*--paper:#10191e/);
  const sizes = [...style.matchAll(/font-size:(\d+)px/g)].map(m => Number(m[1]));
  assert.ok(Math.min(...sizes) >= 13, `smallest font-size ${Math.min(...sizes)}px`);
  // No hard-coded colours remain in inline markup or script-generated HTML.
  assert.doesNotMatch(script, /#0d9488'\:'#cf222e/);
});
