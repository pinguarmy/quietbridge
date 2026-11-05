const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const htmlPath = path.join(__dirname, '..', 'lab.html');
const rawHtml = fs.readFileSync(htmlPath, 'utf8');
const scriptMatch = rawHtml.match(/<script>([\s\S]*?)<\/script>/);

test('lab.html inline script parses cleanly with vm.Script', () => {
  assert.ok(scriptMatch, 'script tag should exist in lab.html');
  const scriptText = scriptMatch[1];
  assert.doesNotThrow(() => {
    new vm.Script(scriptText);
  }, 'Inline script must parse cleanly without syntax errors');
});

test('stepper labels and stages adhere to simplified copy while preserving split semantics', () => {
  // Check HTML stepper markup
  assert.match(rawHtml, /id="stepSetup"[^>]*>1\.\s*Choose phrases<\/button>/);
  assert.match(rawHtml, /id="stepTrain"[^>]*>2\.\s*Practice<\/button>/);
  assert.match(rawHtml, /id="stepValidation"[^>]*>3\.\s*Check on another occasion<\/button>/);
  assert.match(rawHtml, /id="stepTest"[^>]*>4\.\s*Final check<\/button>/);
  assert.match(rawHtml, /id="stepModels"[^>]*>5\.\s*Activate<\/button>/);

  // Check data-stage attributes
  assert.match(rawHtml, /data-stage="setup"/);
  assert.match(rawHtml, /data-stage="train"/);
  assert.match(rawHtml, /data-stage="validation"/);
  assert.match(rawHtml, /data-stage="test"/);
  assert.match(rawHtml, /data-stage="models"/);

  // Check split names in JS
  const scriptText = scriptMatch[1];
  assert.match(scriptText, /const split=currentStage==='setup'\?'setup':\(currentStage==='validation'\?'validation':\(currentStage==='test'\?'test':'train'\)\)/);
});

test('return link is named Back to message (not Live Demo) and validates loopback safely', () => {
  // Link label check
  assert.ok(!rawHtml.includes('Live Demo'), 'Should not contain Live Demo text');
  assert.match(rawHtml, /id="returnLink"[^>]*>Back to message<\/a>/);

  // Extract getSafeReturnUrl function
  const scriptText = scriptMatch[1];
  const fnMatch = scriptText.match(/function getSafeReturnUrl\(\)\{[\s\S]*?\n\}/);
  assert.ok(fnMatch, 'getSafeReturnUrl function must exist');

  function evaluateReturnUrl(queryString, origin = 'http://127.0.0.1:8766') {
    const sandbox = {
      window: {
        location: {
          search: queryString,
          href: `${origin}/lab.html${queryString}`,
          origin,
          port: new URL(origin).port
        }
      },
      URL,
      URLSearchParams
    };
    vm.createContext(sandbox);
    vm.runInContext(fnMatch[0] + '; result = getSafeReturnUrl();', sandbox);
    return sandbox.result;
  }

  // 1. Default when no return_to provided
  assert.equal(evaluateReturnUrl(''), 'http://127.0.0.1:8765/');

  // 2. Safe loopback return_to
  assert.equal(evaluateReturnUrl('?return_to=http%3A%2F%2F127.0.0.1%3A8765%2F'), 'http://127.0.0.1:8765/');
  assert.equal(evaluateReturnUrl('?return_to=http%3A%2F%2Flocalhost%3A9000%2F'), 'http://localhost:9000/');

  // 3. Reject non-loopback domain (e.g. remote attacker)
  assert.equal(evaluateReturnUrl('?return_to=http%3A%2F%2Fexample.com%2F'), 'http://127.0.0.1:8765/');
  assert.equal(evaluateReturnUrl('?return_to=https%3A%2F%2Fgoogle.com%2F'), 'http://127.0.0.1:8765/');

  // 4. Reject non-root path or search/hash
  assert.equal(evaluateReturnUrl('?return_to=http%3A%2F%2F127.0.0.1%3A8765%2Fother'), 'http://127.0.0.1:8765/');
  assert.equal(evaluateReturnUrl('?return_to=http%3A%2F%2F127.0.0.1%3A8765%2F%3Ffoo%3Dbar'), 'http://127.0.0.1:8765/');
  assert.equal(evaluateReturnUrl('?return_to=http%3A%2F%2F127.0.0.1%3A8765%2F%23hash'), 'http://127.0.0.1:8765/');

  // Reject the lab's actual port, including an ephemeral port.
  assert.equal(evaluateReturnUrl('?return_to=http%3A%2F%2F127.0.0.1%3A8766%2F'), 'http://127.0.0.1:8765/');
  assert.equal(evaluateReturnUrl('?return_to=http%3A%2F%2Flocalhost%3A43210%2F', 'http://127.0.0.1:43210'), 'http://127.0.0.1:8765/');
  assert.equal(evaluateReturnUrl('?return_to=http%3A%2F%2Flocalhost%3A8766%2F', 'http://127.0.0.1:43210'), 'http://localhost:8766/');
});

test('first-time user with zero samples is guided to setup without auto-opening camera', () => {
  const scriptText = scriptMatch[1];
  const hasRecordedMatch = scriptText.match(/function hasRecordedSamples\(profileId\)\{[\s\S]*?\n\}/);
  assert.ok(hasRecordedMatch, 'hasRecordedSamples helper should exist');

  const sandbox = {
    progressStore: {},
    allProfiles: [],
    activeProfile: null,
    hasRecordedSamples: null
  };
  vm.createContext(sandbox);
  vm.runInContext(hasRecordedMatch[0] + '; hasRecordedSamples = hasRecordedSamples;', sandbox);

  // When progressStore is empty for profile 'user-1'
  assert.equal(sandbox.hasRecordedSamples('user-1'), false);

  // When counts exist but are all 0
  sandbox.progressStore['user-1:train:gestures'] = { index: 0, counts: [0, 0] };
  assert.equal(sandbox.hasRecordedSamples('user-1'), false);

  // When at least 1 count > 0 in local progressStore
  sandbox.progressStore['user-1:train:gestures'] = { index: 0, counts: [0, 1] };
  assert.equal(sandbox.hasRecordedSamples('user-1'), true);

  // When progressStore is empty but server profile.stats has samples > 0
  sandbox.progressStore = {};
  sandbox.allProfiles = [{
    id: 'user-2',
    stats: {
      sequences: { train: 3, validation: 0, test: 0 },
      mouths: { train: 0, validation: 0, test: 0 }
    }
  }];
  assert.equal(sandbox.hasRecordedSamples('user-2'), true);

  // Check onboarding DOM element in HTML
  assert.match(rawHtml, /id="setupOnboarding"/);
  assert.match(rawHtml, /Choose your personal phrases/);
});

test('advanced technical details, session id, and fps stats are collapsed in <details>', () => {
  assert.match(rawHtml, /<details id="advancedSettings" class="advanced-details">/);
  assert.match(rawHtml, /<code id="sessionLabel">s1<\/code>/);
  assert.match(rawHtml, /<button id="btnNewSession"/);
  assert.match(rawHtml, /<div class="fps-bar" id="fpsBar">/);
  assert.match(rawHtml, /<strong id="measuredFps">/);
  assert.match(rawHtml, /<strong id="receivedFrames">/);

  // Recording status is visible outside details
  assert.match(rawHtml, /id="recordingStatePill"/);
  assert.match(rawHtml, /id="stageName"/);
  assert.match(rawHtml, /id="status"/);
  assert.match(rawHtml, /id="progress"/);
});

test('model training buttons show exactly one per selected model type and evaluateRecognition click is bound', () => {
  const scriptText = scriptMatch[1];

  // Verify evaluateRecognition click binding
  assert.match(scriptText, /\$\('evaluateRecognition'\)\.onclick\s*=\s*\(\)\s*=>/);
  assert.match(scriptText, /modelAction\('evaluate',\s*\{kind:\s*'language',\s*candidate_id\}\)/);

  // Test modelControls visibility logic in sandbox
  const fnMatch = scriptText.match(/function modelControls\(\)\{[\s\S]*?\n\}/);
  assert.ok(fnMatch, 'modelControls function must exist');

  const elements = {
    modelKind: { value: 'motion' },
    candidateSelect: { value: 'cand-1' },
    trainMotion: { hidden: false, disabled: false },
    trainVisual: { hidden: false, disabled: false },
    trainLanguage: { hidden: false, disabled: false },
    refreshModels: { disabled: false },
    rollbackModel: { disabled: false },
    evaluateRecognition: { hidden: false, disabled: false },
    viewReport: { disabled: false },
    promoteModel: { disabled: false }
  };

  const sandbox = {
    $: id => elements[id],
    recording: false,
    working: false,
    modelBusy: false,
    token: 'test-token',
    modelData: {
      candidates: [
        { id: 'cand-1', kind: 'motion', status: 'completed', promotable: true },
        { id: 'cand-2', kind: 'language', status: 'completed', eval_status: 'idle', promotable: true }
      ]
    }
  };
  vm.createContext(sandbox);
  vm.runInContext(fnMatch[0], sandbox);

  // 1. When modelKind is 'motion'
  elements.modelKind.value = 'motion';
  elements.candidateSelect.value = 'cand-1';
  sandbox.modelControls();
  assert.equal(elements.trainMotion.hidden, false, 'trainMotion should be visible');
  assert.equal(elements.trainVisual.hidden, true, 'trainVisual should be hidden');
  assert.equal(elements.trainLanguage.hidden, true, 'trainLanguage should be hidden');
  assert.equal(elements.evaluateRecognition.hidden, true, 'evaluateRecognition should be hidden for motion');

  // 2. When modelKind is 'lips'
  elements.modelKind.value = 'lips';
  sandbox.modelControls();
  assert.equal(elements.trainMotion.hidden, true, 'trainMotion should be hidden');
  assert.equal(elements.trainVisual.hidden, false, 'trainVisual should be visible');
  assert.equal(elements.trainLanguage.hidden, true, 'trainLanguage should be hidden');

  // 3. When modelKind is 'language'
  elements.modelKind.value = 'language';
  elements.candidateSelect.value = 'cand-2';
  sandbox.modelControls();
  assert.equal(elements.trainMotion.hidden, true, 'trainMotion should be hidden');
  assert.equal(elements.trainVisual.hidden, true, 'trainVisual should be hidden');
  assert.equal(elements.trainLanguage.hidden, false, 'trainLanguage should be visible');
  assert.equal(elements.evaluateRecognition.hidden, false, 'evaluateRecognition should be visible for language');
  assert.equal(elements.evaluateRecognition.disabled, false, 'evaluateRecognition should be enabled when candidate completed');

  // 4. When candidate eval_status is 'running'
  sandbox.modelData.candidates[1].eval_status = 'running';
  sandbox.modelControls();
  assert.equal(elements.evaluateRecognition.disabled, true, 'evaluateRecognition should be disabled while running');
});

test('saving a model directs users to reload instead of claiming live activation', async () => {
  const scriptText = scriptMatch[1];
  const source = scriptText.match(/async function modelAction\(path,data\)\{[\s\S]*?\n\}/)[0];
  const notice = { replaceChildren(...children) { this.children = children; } };
  const context = {
    recording: false, working: false, modelBusy: false,
    controls() {}, renderModels() {}, action: async () => ({ active: { language: '/saved/lm.pth' } }),
    $: id => id === 'modelSaveNotice' ? notice : {},
    document: { createElement: () => ({}) },
    URL, getSafeReturnUrl: () => 'http://127.0.0.1:9000/',
  };
  vm.createContext(context);
  vm.runInContext(source, context);
  await context.modelAction('promote', { kind: 'language' });
  assert.match(notice.children[0].textContent, /has not reloaded/);
  assert.equal(notice.children[1].href, 'http://127.0.0.1:9000/settings');
});

test('a11y: 44px min-height buttons, table scrolling container, and reduced motion styles exist', () => {
  // 44px minimum touch target height
  assert.match(rawHtml, /min-height:\s*44px/);

  // Focus visible styles
  assert.match(rawHtml, /:focus-visible/);

  // prefers-reduced-motion media query
  assert.match(rawHtml, /@media\s*\(\s*prefers-reduced-motion:\s*reduce\s*\)/);

  // Horizontal scroll containment for tables to avoid viewport blowouts
  assert.match(rawHtml, /class="table-scroll"/);
  assert.match(rawHtml, /overflow-x:\s*auto/);
});
