const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function screen(page = 'index.html', sessionValues = new Map(), localValues = new Map()) {
  const elements = new Map();
  function element(tag = 'DIV') {
    const classes = new Set();
    return {
      tagName: tag.toUpperCase(), textContent: '', hidden: false, disabled: false,
      value: '', checked: false, style: {}, dataset: {}, children: [], attributes: {},
      classList: { toggle(name, enabled) { if (enabled) classes.add(name); else classes.delete(name); }, contains(name) { return classes.has(name); } },
      setAttribute(name, value) { this.attributes[name] = value; },
      removeAttribute(name) { delete this.attributes[name]; },
      listeners: {},
      matches() { return false; }, addEventListener(name, fn) { this.listeners[name] = fn; }, getContext() { return null; },
      focus() { this.focused = true; }, select() {},
      append(child) { this.children.push(child); if (child.id) elements.set(child.id, child); },
      before(child) { if (child.id) elements.set(child.id, child); },
      replaceChildren(...children) { this.children = children; },
    };
  }
  const html = fs.readFileSync(path.join(__dirname, '../ui/', page), 'utf8');
  for (const [, tag, id] of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"/g)) elements.set(id, element(tag));
  const context = {
    document: {
      getElementById(id) { return elements.get(id) || null; },
      createElement: element, addEventListener() {},
      querySelectorAll(selector) { return selector === '.alternative' ? elements.get('alternatives').children : []; },
    },
    window: { addEventListener() {}, confirm: () => true,
      sessionStorage: {getItem: key => sessionValues.get(key) || null, setItem: (key,value) => sessionValues.set(key,value), removeItem: key => sessionValues.delete(key)},
      localStorage: {getItem: key => localValues.get(key) || null, setItem: (key,value) => localValues.set(key,value)} },
    fetch: () => new Promise(() => {}), setTimeout() {}, clearTimeout() {},
    Date, Promise, console,
    Option: function (label, value) { return { textContent: label, value }; },
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
  return { context, elements, render, sessionValues, localValues };
}

test('reloading models targets the running profile and reports warmup truthfully', async () => {
  const { elements, context, render } = screen('settings.html');
  render({ profile_id: 'alice', profile_name: 'Alice' });
  elements.get('profile-select').value = 'bob';
  const requests = [];
  context.fetch = async (url, options) => {
    requests.push([url, JSON.parse(options.body)]);
    return { ok: true, json: async () => ({ ...context.sample, revision: 2, lips: 'warming', signs: 'loading', profile: { name: 'Alice' } }) };
  };
  await elements.get('profile-reload-btn').onclick();
  assert.deepEqual(requests, [['/api/profile/select', { id: 'alice' }]]);
  assert.equal(elements.get('lip-model').textContent, 'Loading…');
  assert.match(elements.get('profile-switch-status').textContent, /loading/);
  assert.equal(elements.get('profile-reload-btn').disabled, false);
});

test('isolated sign backend retains shared-window capture controls', () => {
  const { elements, render } = screen();
  render({ sign_input: 'isolated', session_active: true, auto_recognition: true });
  assert.equal(elements.get('record').hidden, false);
  assert.equal(elements.get('record').disabled, false);
});

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

test('removed AI alternative is no longer selected after disable', () => {
  const { elements, render } = screen();
  const decision = { text: 'Please wait', status: 'confirmation_needed', reason: 'lip_only' };
  render({ decision, alternatives: ['Please wait', 'Please wait.'], cleanup: { status: 'ready', suggestion: 'Please wait.', source_revision: 1 } });
  elements.get('alternatives').children[1].onclick();
  assert.equal(elements.get('message').textContent, 'Please wait.');
  render({ decision, alternatives: ['Please wait'], cleanup: { status: 'disabled' } });
  assert.equal(elements.get('message').textContent, 'Please wait');
});

test('sign vocabulary refreshes when a slow model finishes loading', () => {
  const { elements, render } = screen('settings.html');
  render({ vocabulary: {}, signs: 'loading' });
  assert.equal(elements.get('labels').children.length, 0);
  render({ vocabulary: { BSL_WATER: { phrase: 'water', matches: ['water'], enabled: true, verified: false } }, signs: 'ready', sign_input: 'isolated', sign_model: { language: 'bsl', sha256: 'a'.repeat(64) } });
  assert.equal(elements.get('labels').children.length, 1);
  assert.match(elements.get('sign-model-status').textContent, /BSL isolated/);
});

test('choosing a conflict alternative explicitly replaces an unfinished draft and never applies stale text', async () => {
  const a=screen();a.render({profile_id:'alice'});
  a.elements.get('corrected-text').value='Old unfinished draft';a.elements.get('corrected-text').listeners.input();
  a.render({profile_id:'alice',revision:2,decision:{text:'I need water',status:'confirmation_needed',reason:'conflict'},alternatives:['I need water','No.']});
  a.context.window.confirm=()=>false;
  a.elements.get('alternatives').children[1].onclick();
  assert.equal(a.elements.get('corrected-text').value,'Old unfinished draft');
  assert.equal(a.elements.get('speak').disabled,true);
  a.context.window.confirm=()=>true;
  a.elements.get('alternatives').children[1].onclick();
  assert.equal(a.elements.get('message').textContent,'No.');
  assert.equal(a.elements.get('corrected-text').value,'No.');
  assert.equal(a.sessionValues.size,0);
  const calls=[];
  a.context.fetch=async(url,options)=>{calls.push(JSON.parse(options.body));return {ok:true,json:async()=>({...a.context.sample,revision:3,decision:{text:'No.',status:'accepted',reason:'corrected'}})}};
  await a.elements.get('save-correction').onclick();
  assert.equal(calls[0].text,'No.');
  assert.equal(a.elements.get('message').textContent,'No.');
});

test('main page shows backend input failures while keeping typing available', () => {
  const a=screen();a.render({camera_error:'Camera stopped',sign_error:'Model missing'});
  assert.match(a.elements.get('input-warning').textContent,/Camera stopped.*Model missing.*type your message/);
  assert.equal(a.elements.get('corrected-text').disabled,false);
});

test('unfinished editor survives new revisions and a fresh page in the same tab', () => {
  const storage = new Map();
  const a = screen('index.html', storage); a.render({profile_id:'alice'});
  a.elements.get('corrected-text').value = 'My unfinished words';
  a.elements.get('corrected-text').listeners.input();
  a.render({profile_id:'alice',revision:2,decision:{text:'New recognition',status:'accepted',reason:'lip_only'}});
  assert.equal(a.elements.get('corrected-text').value,'My unfinished words');
  assert.equal(a.elements.get('speak').disabled,true);
  const b = screen('index.html', storage); b.render({profile_id:'alice',decision:{text:'Backend words',status:'accepted',reason:'lip_only'}});
  assert.equal(b.elements.get('corrected-text').value,'My unfinished words');
  assert.equal(b.elements.get('speak').disabled,true);
});

test('record control toggles with one activation, without sustained hold', async () => {
  const a=screen(); a.render(); const calls=[];
  a.context.fetch=async(url,options)=>{
    const action=url.split('/').pop(); calls.push(action);
    return {ok:true,json:async()=>({...a.context.sample,phase:action==='start'?'recording':'tail'})};
  };
  await a.elements.get('record').onclick();
  assert.equal(a.elements.get('record-label').textContent,'Finish recording');
  await a.elements.get('record').onclick();
  assert.deepEqual(calls,['start','stop']);
});

test('guide is dismissible and reopenable without changing message state', () => {
  const a=screen(); a.render();
  assert.equal(a.elements.get('first-use-guide').hidden,false);
  a.elements.get('guide-dismiss').onclick();
  assert.equal(a.elements.get('first-use-guide').hidden,true);
  a.elements.get('open-guide').onclick();
  assert.equal(a.elements.get('first-use-guide').hidden,false);
  assert.equal(a.elements.get('guide-heading').focused,true);
  assert.equal(a.elements.get('message').textContent,'Your words will appear here.');
});

test('explicit new message sends current revision and clears unfinished editor only on success', async () => {
  const a=screen(); a.render({profile_id:'alice',decision:{text:'Previous',status:'accepted',reason:'corrected'}});
  a.elements.get('corrected-text').value='Unfinished';a.elements.get('corrected-text').listeners.input();
  let body;
  a.context.fetch=async(url,options)=>{body=JSON.parse(options.body);return {ok:true,json:async()=>({...a.context.sample,revision:2,decision:{text:'',status:'unavailable',reason:'lip_only'}})}};
  await a.elements.get('new-message').onclick();
  assert.deepEqual(body,{revision:1});
  assert.equal(a.elements.get('corrected-text').value,'');
  assert.equal(a.sessionValues.size,0);
  assert.equal(a.elements.get('speak').disabled,true);
});

test('failed apply keeps unfinished text and blocks stale speech', async () => {
  const a=screen();a.render({profile_id:'alice',decision:{text:'Old words',status:'accepted',reason:'corrected'}});
  a.elements.get('corrected-text').value='New words';a.elements.get('corrected-text').listeners.input();
  a.context.fetch=async()=>{throw Error('Network unavailable')};
  await a.elements.get('save-correction').onclick();
  assert.equal(a.elements.get('corrected-text').value,'New words');
  assert.equal(a.elements.get('speak').disabled,true);
  assert.match(a.elements.get('error').textContent,/edit is kept/);
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
  elements.get('corrected-text').listeners.input();
  await elements.get('save-correction').onclick();
  await vm.runInContext('actionChain', context);
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0], ['/api/action/correct', { revision: 1, text: 'Please wait', learn: false }]);
  assert.equal(elements.get('message').textContent, 'Please wait');
  assert.equal(elements.get('speak').disabled, false);
  assert.equal(elements.get('save-correction').disabled, true);
});
