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

test('page uses the shared design system: linked stylesheet, tokens only, dark mode, 14px type floor', () => {
  assert.match(html, /<link rel="stylesheet" href="\/design.css">/);
  assert.match(html, /<link rel="icon" href="\/favicon.svg" type="image\/svg\+xml">/);
  // The page's own styles add layout only: no hard-coded colours (design.css owns light and dark themes).
  const colours = style.match(/(?:^|[:\s,(])#[0-9a-fA-F]{3,8}(?![\w-])|rgba?\(|hsla?\(/g) || [];
  assert.deepEqual(colours, []);
  assert.match(style, /var\(--accent-fill\)/);
  assert.match(style, /var\(--danger-fill\)/);
  // No local colour-scheme override that would break the shared dark theme.
  assert.doesNotMatch(style, /prefers-color-scheme/);
  const sizes = [...style.matchAll(/font-size:\s*(\d+)px/g)].map(m => Number(m[1]));
  assert.ok(sizes.every(size => size >= 14), `smallest font-size ${Math.min(...sizes)}px`);
  assert.doesNotMatch(style, /var\(--text-(?:2xs|xxs)\)/);
  // No inline style attributes in markup or script-generated HTML.
  assert.doesNotMatch(html.replace(/<style>[\s\S]*?<\/style>/, ''), /\sstyle="/);
  assert.match(style, /@media \(forced-colors:active\)/);
  // Live recording indicator pulses only when the person has not asked for reduced motion.
  const pulse = style.indexOf('animation:live-pulse');
  assert.ok(pulse > style.indexOf('@media (prefers-reduced-motion:no-preference)'));
  assert.ok(pulse < style.indexOf('@media (prefers-reduced-motion:reduce)'));
});

test('recorded steps are marked from saved tries only, with a text alternative', () => {
  const source = script.match(/function stageHasRecordings\(stage\)\{[\s\S]*?\n\}/)[0];
  const context = {activeProfile: {id: 'p'}, allProfiles: [{id: 'p', stats: {sequences: {train: 2, validation: 0, test: 0}, mouths: {train: 0, validation: 0, test: 0}}}],
    progressStore: {'p:test:lips': {index: 0, counts: [0, 1]}, 'other:validation:gestures': {index: 0, counts: [4]}}};
  vm.createContext(context); vm.runInContext(source, context);
  assert.equal(context.stageHasRecordings('train'), true);
  assert.equal(context.stageHasRecordings('validation'), false);
  assert.equal(context.stageHasRecordings('test'), true);
  // Choosing phrases and activating are never shown as "recorded".
  assert.equal(context.stageHasRecordings('setup'), false);
  assert.equal(context.stageHasRecordings('models'), false);
  context.activeProfile = null;
  assert.equal(context.stageHasRecordings('train'), false);
  const render = script.slice(script.indexOf('function render(){'), script.indexOf('function checkStageCompletion()'));
  assert.match(render, /classList\.toggle\('done',recorded\)/);
  assert.match(render, /' \(has recordings\)'/);
  assert.equal((html.match(/<span class="step-state sr-only"><\/span>/g) || []).length, 5);
});

test('record control is one labelled toggle and video consent is stated next to it', () => {
  const controls = script.match(/function controls\(\)\{[\s\S]*?\n\}/)[0];
  assert.match(controls, /recording\?'Finish and save':'Start recording'/);
  assert.match(controls, /videoRequested\(\)&&\(!\$\('videoConsent'\)\.checked\|\|!signModelInfo\)/);
  assert.ok(!html.includes('Finish &'), 'one spelling of the finish label');
  const box = html.slice(html.indexOf('id="videoConsentBox"'), html.indexOf('id="record"'));
  assert.match(box, /Recording stays off until you agree/);
  assert.match(box, /id="videoConsent" type="checkbox">/);
  assert.ok(html.indexOf('id="videoConsentBox"') < html.indexOf('id="record"'), 'consent comes before the record button');
});

test('phrase and gesture rows are list items with labelled controls', () => {
  assert.match(html, /<ul class="gesture-list gesture-editor" id="gestureTbody" aria-label="Personal gesture definitions"><\/ul>/);
  assert.match(html, /<ol class="phrase-list" id="promptTbody" aria-label="Phrases to mouth silently"><\/ol>/);
  const render = script.slice(script.indexOf('function renderSetupView()'), script.indexOf('function unusedGestureLabel('));
  assert.match(render, /aria-label="Practice phrase \$\{idx\+1\}"/);
  assert.match(render, /aria-label="Remove phrase \$\{idx\+1\}"/);
  assert.match(render, /aria-label="Remove gesture: /);
  // Icons are decorative and never take focus.
  for (const svg of html.match(/<svg[^>]*>/g)) assert.match(svg, /aria-hidden="true" focusable="false"/, svg);
  // Removing a gesture moves focus to a row summary, not to a nested "Advanced" summary.
  assert.match(render, /querySelectorAll\('details\.gesture-definition > summary'\)/);
  assert.doesNotMatch(script, /createElement\('tr'\);\s*tr\.className='(?:gesture|phrase)-row'/);
});

test('opening Activate returns keyboard focus to the step after status loads', async () => {
  const source = script.match(/function setStage\(stage\)\{[\s\S]*?\n\}/)[0];
  let focused = null;
  const step = {classList: {contains: c => c === 'step-btn'}, focus() { focused = 'stepModels'; }};
  const document = {activeElement: step, body: {}};
  const context = {document, recording: false, working: false, modelBusy: false, currentStage: 'train',
    confirmDiscardSetup: () => true, render() {}, refreshModelStatus: async () => { document.activeElement = document.body; }};
  vm.createContext(context); vm.runInContext(source, context);
  context.setStage('models');
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.equal(focused, 'stepModels');
});
