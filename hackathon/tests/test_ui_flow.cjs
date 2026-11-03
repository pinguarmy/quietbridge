const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function screen() {
  const elements = new Map();
  function element(tag = 'DIV') {
    const classes = new Set();
    return {
      tagName: tag.toUpperCase(), textContent: '', hidden: false, disabled: false,
      value: '', checked: false, style: {}, dataset: {}, children: [], attributes: {},
      classList: { toggle(name, enabled) { if (enabled) classes.add(name); else classes.delete(name); }, contains(name) { return classes.has(name); } },
      setAttribute(name, value) { this.attributes[name] = value; },
      removeAttribute(name) { delete this.attributes[name]; },
      matches() { return false; }, addEventListener() {}, getContext() { return null; },
      append(child) { this.children.push(child); if (child.id) elements.set(child.id, child); },
      before(child) { if (child.id) elements.set(child.id, child); },
      replaceChildren(...children) { this.children = children; },
    };
  }
  const html = fs.readFileSync(path.join(__dirname, '../ui/index.html'), 'utf8');
  for (const [, tag, id] of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"/g)) elements.set(id, element(tag));
  const context = {
    document: {
      getElementById(id) { return elements.get(id) || null; },
      createElement: element, addEventListener() {},
      querySelectorAll(selector) { return selector === '.alternative' ? elements.get('alternatives').children : []; },
    },
    window: { addEventListener() {} },
    fetch: () => new Promise(() => {}), setTimeout() {}, clearTimeout() {},
    Date, Promise, console,
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../ui/app.js'), 'utf8'), context);
  function render(overrides = {}) {
    context.sample = {
      revision: 1, rehearsal: false, browser_camera: true, camera: 'ready', lips: 'ready', signs: 'ready',
      sign_input: 'stock', mode: 'combined', phase: 'idle', speech_available: true,
      vocabulary: {}, metrics: {}, alternatives: [], session_text: '', session_active: false,
      session_finalizing: false, auto_recognition: false, cleanup: { status: 'disabled' },
      decision: { text: '', status: 'unavailable', reason: 'lip_only' }, ...overrides,
    };
    vm.runInContext('render(sample)', context);
  }
  return { context, elements, render };
}

test('custom gestures retain a usable record control inside a camera session', () => {
  const { elements, render } = screen();
  render({ sign_input: 'motion', session_active: true, auto_recognition: true });
  assert.equal(elements.get('record').hidden, false);
  assert.equal(elements.get('record').disabled, false);
  assert.match(elements.get('record-label').textContent, /lips \+ gesture/);
  assert.match(elements.get('entry-hint').textContent, /together/);
  render({ mode: 'lips', sign_input: 'motion', session_active: true, auto_recognition: true });
  assert.equal(elements.get('record').hidden, true);
});

test('conflicts require a choice and AI wording never replaces the selected message', () => {
  const { elements, render } = screen();
  render({ decision: { text: 'I do not need water', status: 'confirmation_needed', reason: 'conflict' }, alternatives: ['I do not need water', 'I need water'] });
  assert.equal(elements.get('speak').disabled, true);
  elements.get('alternatives').children[0].onclick();
  assert.equal(elements.get('message').textContent, 'I do not need water');
  assert.equal(elements.get('speak').disabled, false);
  render({ revision: 2, decision: { text: 'Original words', status: 'confirmation_needed', reason: 'lip_only' }, alternatives: ['Original words', 'AI words'], cleanup: { status: 'ready', suggestion: 'AI words', source_revision: 2 } });
  assert.equal(elements.get('message').textContent, 'Original words');
  assert.match(elements.get('alternatives').children[1].textContent, /AI suggestion/);
  elements.get('alternatives').children[1].onclick();
  assert.equal(elements.get('message').textContent, 'AI words');
});

test('typed messages use the current revision without speaking or enabling learning', async () => {
  const { elements, context, render } = screen();
  render();
  const calls = [];
  context.fetch = async (url, options) => {
    calls.push([url, JSON.parse(options.body)]);
    return { ok: true, json: async () => ({ ...context.sample, revision: 2, alternatives: ['Please wait'], decision: { text: 'Please wait', status: 'accepted', reason: 'corrected' } }) };
  };
  elements.get('corrected-text').value = 'Please wait';
  elements.get('save-correction').onclick();
  await vm.runInContext('actionChain', context);
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0], ['/api/action/correct', { revision: 1, text: 'Please wait', learn: false }]);
  assert.equal(elements.get('message').textContent, 'Please wait');
  assert.equal(elements.get('speak').disabled, false);
  assert.equal(elements.get('save-correction').disabled, true);
});
