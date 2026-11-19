const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '..', 'lab.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const fn = name => script.match(new RegExp(`function ${name}\\([^)]*\\)\\{[\\s\\S]*?\\n\\}`))[0];

function modelSandbox(modelData) {
  const ids = ['trainMotion','trainVisual','trainLanguage','refreshModels','rollbackModel','modelKind','candidateSelect',
    'evaluateRecognition','viewReport','promoteModel','restoreHint'];
  const elements = Object.fromEntries(ids.map(id => [id, {disabled: false, hidden: false, textContent: ''}]));
  elements.modelKind.value = 'motion';
  elements.modelKind.selectedOptions = [{textContent: 'Gestures (motion)'}];
  elements.candidateSelect.value = '';
  const sandbox = {$: id => elements[id], recording: false, working: false, modelBusy: false, token: 't', modelData};
  vm.createContext(sandbox);
  vm.runInContext(fn('modelControls') + fn('restoreHint'), sandbox);
  return {sandbox, elements};
}

test('restore is disabled and explained when the server reports no rollback point', () => {
  const {sandbox, elements} = modelSandbox({candidates: [], restore: {motion: {available: false, restores: null}}});
  sandbox.modelControls();
  assert.equal(elements.rollbackModel.disabled, true);
  assert.match(elements.restoreHint.textContent, /Nothing to restore/);
});

test('restore explains whether it returns to a saved model or to the standard behaviour', () => {
  const {sandbox, elements} = modelSandbox({candidates: [], restore: {motion: {available: true, restores: 'stock'}}});
  sandbox.modelControls();
  assert.equal(elements.rollbackModel.disabled, false);
  assert.match(elements.restoreHint.textContent, /no saved Gestures \(motion\) model/);
  sandbox.modelData.restore.motion.restores = 'previous_model';
  sandbox.modelControls();
  assert.match(elements.restoreHint.textContent, /saved before the current one/);
});

test('restore stays disabled when the server does not report restore data at all', () => {
  const {sandbox, elements} = modelSandbox({candidates: []});
  sandbox.modelControls();
  assert.equal(elements.rollbackModel.disabled, true);
  assert.match(elements.restoreHint.textContent, /unavailable/);
});

test('new gestures get an unused internal label even after a removal', () => {
  const ctx = {}; vm.createContext(ctx); vm.runInContext(fn('unusedGestureLabel'), ctx);
  // Five gestures, the first removed, then one added: GESTURE_6 already exists.
  const gestures = ['WATER', 'YES', 'NO', 'REST', 'GESTURE_6'].map(label => ({label}));
  assert.equal(ctx.unusedGestureLabel(gestures), 7);
  assert.equal(ctx.unusedGestureLabel([{label: 'HELLO'}]), 2);
});

test('profile save errors are explained in page terms, not server field names', () => {
  const ctx = {}; vm.createContext(ctx); vm.runInContext(fn('setupErrorHelp'), ctx);
  assert.match(ctx.setupErrorHelp("Duplicate gesture label 'GESTURE_6'"), /same internal label GESTURE_6.*Advanced/);
  assert.match(ctx.setupErrorHelp('lip_prompts must be a non-empty list of strings'), /at least one phrase/);
  assert.match(ctx.setupErrorHelp('Lip prompt must be 1-300 characters'), /1 to 300 characters/);
  assert.doesNotMatch(ctx.setupErrorHelp('lip_prompts must be a non-empty list of strings'), /lip_prompts/);
});

test('report summary shows unmeasured results as not measured, never as zero', () => {
  const ctx = {}; vm.createContext(ctx);
  vm.runInContext(fn('ratio') + fn('pct') + fn('reportRows'), ctx);
  const rows = Object.fromEntries(ctx.reportRows({kind: 'motion', report: {
    train_trials: 8, validation_trials: 4, test_trials: 0,
    validation_ungated: {trials: 4, positive_trials: 3, positive_correct: 2, rest_trials: 0, rest_correctly_silent: 0, false_positives: 1},
    test_ungated: {trials: 0, positive_trials: 0, positive_correct: 0, rest_trials: 0, rest_correctly_silent: 0, false_positives: 0}}}));
  assert.equal(rows['Another occasion: gestures recognized'], '2 of 3');
  assert.match(rows['Another occasion: rest stayed silent'], /not measured/);
  assert.match(rows['Final check: gestures recognized'], /not measured/);
  assert.match(rows['Final check: wrong gesture or false alarm'], /not measured/);
  const lips = Object.fromEntries(ctx.reportRows({kind: 'lips', report: {counts: {train: 3, validation: 2, test: 0},
    baseline: {val: {metrics: {wer: 0.5}}, test: {metrics: {wer: null}}}, candidate: {val: {metrics: {wer: 0.25}}}}}));
  assert.equal(lips['Another occasion: word error rate, candidate'], '25.0%');
  assert.equal(lips['Final check: word error rate, standard'], 'not measured');
});

test('candidate list and summary do not show raw ids, file names or pipe-joined status', () => {
  const render = fn('renderModels');
  assert.doesNotMatch(render, /model_path\?\.split/);
  assert.doesNotMatch(fn('renderCandidateSummary'), /join\(' \| '\)/);
  assert.match(html, /<details id="reportDetails"><summary>Full report \(technical\)<\/summary>/);
});
