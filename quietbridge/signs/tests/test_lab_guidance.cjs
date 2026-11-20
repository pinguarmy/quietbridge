// Personalize page: capture-quality review and "What to record next" rendering contracts.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'lab.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const fn = name => script.match(new RegExp(`function ${name}\\([^)]*\\)\\{[\\s\\S]*?\\n\\}`))[0];

test('review markup offers both choices, uses design-system parts and no inline styles', () => {
  assert.match(html, /id="qualityReview" class="callout callout-warn quality-review" role="alert"[^>]*hidden/);
  assert.match(html, /id="qualityDiscard"[^>]*>Discard and retry<\/button>/);
  assert.match(html, /id="qualitySave"[^>]*>Save anyway<\/button>/);
  assert.match(html, /Nothing has been saved yet/);
  assert.match(html, /id="guidancePanel" class="card guidance"/);
  assert.match(html, /<dl id="guidanceFacts" class="facts"/);
  assert.match(html, /counts, not results/);
  assert.doesNotMatch(html, /\sstyle="/);
});

test('finishing asks for review first and Save anyway sends the explicit choice', () => {
  const record = script.slice(script.indexOf("$('record').onclick=async()=>"), script.indexOf('function afterSave('));
  assert.match(record, /action\(STOP_PATH\[mode\],\{review:true\}\)/);
  assert.match(record, /if\(result\?\.review_required\)\{[\s\S]*?showQualityReview\(result\.quality\);\s*return;/);
  const save = script.slice(script.indexOf("$('qualitySave').onclick"), script.indexOf("$('qualityDiscard').onclick"));
  assert.match(save, /action\(STOP_PATH\[mode\],\{accept_quality:true\}\)/);
  assert.match(script, /const STOP_PATH=\{gestures:'stop',lips:'mouth\/stop',paired:'paired\/stop'\}/);
  // Discard and retry goes through the normal discard, which saves nothing.
  assert.match(script, /\$\('qualityDiscard'\)\.onclick=\(\)=>\$\('discard'\)\.onclick\(\);/);
  assert.match(script, /b\.disabled=[^;]*reviewPending/);
});

test('only blocking warnings are listed for review; informational ones are kept as notes', () => {
  const items = [];
  const elements = {qualityList: {replaceChildren: (...c) => items.push(...c)}, qualityReview: {hidden: true},
    qualityDiscard: {focus() {}}, recordingMeter: {textContent: 'Recording: 0.7 s'}};
  const ctx = {$: id => elements[id], document: {createElement: () => ({})}, note() {}, controls() {}, reviewPending: false,
    setTimeout: f => f()};
  vm.createContext(ctx);
  vm.runInContext(fn('showQualityReview'), ctx);
  ctx.showQualityReview({warnings: [{severity: 'warn', message: 'Your hand hardly moved.'},
    {severity: 'info', message: 'No hands were visible.'}]});
  assert.deepEqual(items.map(i => i.textContent), ['Your hand hardly moved.']);
  assert.equal(elements.qualityReview.hidden, false);
  assert.equal(elements.recordingMeter.textContent, '');
});

test('guidance is filtered by step and mode; Activate shows everything', () => {
  const ctx = {currentStage: 'train', mode: 'gestures'};
  vm.createContext(ctx);
  vm.runInContext(fn('guidanceStage') + fn('guidanceRelevant'), ctx);
  const item = (area, stage) => ({area, stage});
  assert.equal(ctx.guidanceRelevant(item('gestures', 'train')), true);
  assert.equal(ctx.guidanceRelevant(item('gestures', 'setup')), true);
  assert.equal(ctx.guidanceRelevant(item('gestures', 'validation')), false);
  assert.equal(ctx.guidanceRelevant(item('lips', 'train')), false);
  ctx.currentStage = 'validation'; ctx.mode = 'lips';
  assert.equal(ctx.guidanceRelevant(item('lips', 'validation')), true);
  assert.equal(ctx.guidanceRelevant(item('language', 'validation')), true);
  assert.equal(ctx.guidanceRelevant(item('gestures', 'validation')), false);
  ctx.currentStage = 'models';
  assert.equal(ctx.guidanceRelevant(item('gestures', 'test')), true);
});

test('each suggestion renders its finding and next step as text, with a severity badge', () => {
  const make = () => ({append(...c) { this.children = c; }, set className(v) { this._c = v; }, get className() { return this._c; }});
  const ctx = {document: {createElement: make}};
  vm.createContext(ctx);
  vm.runInContext("const GUIDANCE_BADGE={action:['badge-warn','Needed first'],warn:['badge-accent','Suggested'],info:['','Good to know']};" + fn('guidanceItem'), ctx);
  const li = ctx.guidanceItem({severity: 'warn', finding: 'Hello was recognised in 2 of 5 check tries.', next_step: 'Record 5 more practice tries.'});
  const [badge, body] = li.children;
  assert.equal(badge.className, 'badge badge-accent');
  assert.equal(badge.textContent, 'Suggested');
  assert.equal(body.children[0].textContent, 'Hello was recognised in 2 of 5 check tries.');
  assert.equal(body.children[1].textContent, 'Next: Record 5 more practice tries.');
  assert.doesNotMatch(fn('guidanceItem'), /innerHTML/);
});
