const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function screen(page = 'index.html', sessionValues = new Map(), localValues = new Map()) {
  const elements = new Map(), timers = new Map(), windowListeners = new Map();
  let timerId = 0;
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
    window: { addEventListener(name, fn) { windowListeners.set(name, fn); }, confirm: () => true,
      sessionStorage: {getItem: key => sessionValues.has(key) ? sessionValues.get(key) : null, setItem: (key,value) => sessionValues.set(key,value), removeItem: key => sessionValues.delete(key)},
      localStorage: {getItem: key => localValues.get(key) || null, setItem: (key,value) => localValues.set(key,value)} },
    fetch: () => new Promise(() => {}),
    setTimeout(fn, ms) { const id=++timerId; timers.set(id,{fn,ms});return id; },
    clearTimeout(id) { timers.delete(id); },
    Date, Promise, console, AbortController,
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
  return { context, elements, render, sessionValues, localValues, timers, windowListeners };
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

test('editing during speech confirmation cancels dispatch until the newer edit is reviewed', async () => {
  const a=screen();a.render({server_instance:'boot',token:'token',decision:{text:'I need water',status:'confirmation_needed',reason:'conflict'},alternatives:['I need water','No.']});
  a.elements.get('alternatives').children[1].onclick();
  let finish, confirming;
  const requested=new Promise(resolve=>{confirming=resolve});const calls=[];
  a.context.fetch=async(url,options)=>{
    calls.push(url);
    if(url.endsWith('/confirm')) {confirming();return new Promise(resolve=>{finish=resolve})}
    return {ok:true,json:async()=>({...a.context.sample,decision:{text:'No.',status:'accepted',reason:'conflict'}})};
  };
  const speaking=a.elements.get('speak').onclick();await requested;
  assert.equal(a.elements.get('corrected-text').disabled,false);
  a.elements.get('corrected-text').value='Wait, I meant yes';a.elements.get('corrected-text').listeners.input();
  finish({ok:true,json:async()=>({...a.context.sample,decision:{text:'No.',status:'accepted',reason:'conflict'}})});
  await speaking;
  assert.deepEqual(calls,['/api/action/confirm']);
  assert.equal(a.elements.get('corrected-text').value,'Wait, I meant yes');
  assert.equal(a.elements.get('speak').disabled,true);
  assert.match(a.elements.get('error').textContent,/changed|edit/i);
});

test('unchanged explicit conflict confirmation still dispatches the chosen words once', async () => {
  const a=screen();a.render({server_instance:'boot',token:'token',decision:{text:'I need water',status:'confirmation_needed',reason:'conflict'},alternatives:['I need water','No.']});
  a.elements.get('alternatives').children[1].onclick();
  const calls=[];
  a.context.fetch=async(url,options)=>{
    calls.push([url,JSON.parse(options.body)]);
    return {ok:true,json:async()=>({...a.context.sample,decision:{text:'No.',status:'accepted',reason:'conflict'}})};
  };
  await a.elements.get('speak').onclick();
  assert.deepEqual(calls,[['/api/action/confirm',{text:'No.',revision:1}],['/api/action/speak',{revision:1,text:'No.'}]]);
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

test('an intentionally emptied draft survives navigation and remains unapplied', () => {
  const storage=new Map();
  const a=screen('index.html',storage);a.render({profile_id:'alice',decision:{text:'Recognized words',status:'accepted',reason:'corrected'}});
  a.elements.get('corrected-text').value='';a.elements.get('corrected-text').listeners.input();
  assert.equal(a.elements.get('speak').disabled,true);
  const b=screen('index.html',storage);b.render({profile_id:'alice',decision:{text:'Recognized words',status:'accepted',reason:'corrected'}});
  assert.equal(b.elements.get('corrected-text').value,'');
  assert.equal(b.elements.get('speak').disabled,true);
  assert.match(b.elements.get('draft-status').textContent,/preserved/);
  b.elements.get('corrected-text').value='Recognized words';b.elements.get('corrected-text').listeners.input();
  assert.equal(storage.size,0);
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

test('server restart accepts the new identity and token without replaying stale speech', async () => {
  const a=screen();a.render({server_instance:'boot-a',token:'old-token',revision:8,decision:{text:'Old accepted words',status:'accepted',reason:'corrected'}});
  a.elements.get('corrected-text').value='Unfinished after restart';a.elements.get('corrected-text').listeners.input();
  a.render({server_instance:'boot-b',token:'new-token',revision:0});
  assert.equal(a.elements.get('message').textContent,'Your words will appear here.');
  assert.equal(a.elements.get('corrected-text').value,'Unfinished after restart');
  assert.equal(a.elements.get('speak').disabled,true);
  assert.match(a.elements.get('error').textContent,/restarted/);
  assert.equal(vm.runInContext('token',a.context),'new-token');
  a.render({server_instance:'boot-a',token:'old-token',revision:9,decision:{text:'Late old result',status:'accepted',reason:'corrected'}});
  assert.equal(vm.runInContext('token',a.context),'new-token');
  assert.equal(a.elements.get('message').textContent,'Your words will appear here.');
  a.render({server_instance:'boot-b',token:'new-token',revision:2,decision:{text:'Current words',status:'accepted',reason:'corrected'}});
  a.render({server_instance:'boot-b',token:'new-token',revision:1,decision:{text:'Out of date',status:'accepted',reason:'corrected'}});
  assert.equal(a.elements.get('message').textContent,'Current words');
});

for (const boundary of ['pagehide', 'restart']) {
  test(`camera permission granted after ${boundary} cannot start a session`, async () => {
    const a=screen();a.render({server_instance:'before',token:'before-token'});
    let grant, stopped=0;const calls=[];
    const track={stop(){stopped++;},addEventListener(){}};
    const stream={getTracks:()=>[track],getVideoTracks:()=>[track]};
    a.context.navigator={mediaDevices:{getUserMedia:()=>new Promise(resolve=>{grant=resolve;})}};
    a.elements.get('browser-camera').play=async()=>{};
    a.context.fetch=async (url,options)=>{calls.push([url,JSON.parse(options.body)]);return{ok:true,json:async()=>({...a.context.sample,revision:2})};};
    const starting=a.elements.get('browser-camera-button').onclick();
    if(boundary==='pagehide')a.windowListeners.get('pagehide')({persisted:true});
    else a.render({server_instance:'after',token:'after-token',revision:0});
    grant(stream);await starting;
    assert.equal(stopped,1,'late granted tracks are stopped');
    assert.equal(calls.length,0,'no old permission continuation starts capture');
    assert.equal(a.elements.get('browser-camera').srcObject,null);
    assert.equal(a.elements.get('browser-camera-button').textContent,'Start camera');
  });
}

test('camera playback completing after page exit cannot request capture', async () => {
  const a=screen();a.render({server_instance:'boot',token:'token'});
  let played,enteredPlay,stopped=0;const calls=[];
  const playReady=new Promise(resolve=>{enteredPlay=resolve;});
  const track={stop(){stopped++;},addEventListener(){}};
  const stream={getTracks:()=>[track],getVideoTracks:()=>[track]};
  a.context.navigator={mediaDevices:{getUserMedia:async()=>stream}};
  a.elements.get('browser-camera').play=()=>{enteredPlay();return new Promise(resolve=>{played=resolve;});};
  a.context.fetch=async(url,options)=>{calls.push(JSON.parse(options.body));return{ok:true,json:async()=>a.context.sample};};
  const starting=a.elements.get('browser-camera-button').onclick();await playReady;
  a.windowListeners.get('pagehide')({persisted:true});played();await starting;
  assert.equal(stopped,1);
  assert.equal(calls.some(body=>body.active===true),false);
  assert.equal(calls.filter(body=>body.active===false).length,1,'best-effort exit notice only');
});

test('unchanged camera start and explicit finish each request once', async () => {
  const a=screen();a.render({server_instance:'boot',token:'token'});
  let stopped=0;const calls=[];
  const track={stop(){stopped++;},addEventListener(){}};
  const stream={getTracks:()=>[track],getVideoTracks:()=>[track]};
  a.context.navigator={mediaDevices:{getUserMedia:async()=>stream}};
  a.elements.get('browser-camera').play=async()=>{};
  a.context.fetch=async(url,options)=>{const body=JSON.parse(options.body);calls.push(body);return{ok:true,json:async()=>({...a.context.sample,revision:a.context.sample.revision+1,session_active:body.active})};};
  await a.elements.get('browser-camera-button').onclick();
  assert.equal(a.elements.get('browser-camera').srcObject,stream);
  assert.equal(a.elements.get('browser-camera-button').textContent,'Finish message');
  await a.elements.get('browser-camera-button').onclick();
  assert.equal(stopped,1);
  assert.deepEqual(calls.map(body=>body.active),[true,false]);
  assert.equal(a.elements.get('browser-camera').srcObject,null);
});

test('an old camera-start rejection cannot replace the new server restart notice', async () => {
  const a=screen();a.render({server_instance:'before',token:'old-token'});
  let rejectStart,started,stopped=0;const requestReady=new Promise(resolve=>{started=resolve;});
  const track={stop(){stopped++;},addEventListener(){}};
  const stream={getTracks:()=>[track],getVideoTracks:()=>[track]};
  a.context.navigator={mediaDevices:{getUserMedia:async()=>stream}};
  a.elements.get('browser-camera').play=async()=>{};
  const requests=[];a.context.fetch=(url,options)=>{requests.push([url,options.headers['X-Lipflow-Token']]);started();return new Promise((resolve,reject)=>{rejectStart=reject;});};
  const starting=a.elements.get('browser-camera-button').onclick();await requestReady;
  a.render({server_instance:'after',token:'new-token',revision:0});
  rejectStart(new Error('Old request failed'));await starting;
  assert.match(a.elements.get('error').textContent,/restarted/);
  assert.equal(stopped,1);
  assert.deepEqual(requests,[['/api/action/camera_session','old-token']]);
});

test('a delayed camera-auth refresh cannot restore a retired server token', async () => {
  const a=screen();a.render({server_instance:'old',token:'old-token',revision:5});
  const old={...a.context.sample};let finish, reached;
  const refreshing=new Promise(resolve=>{reached=resolve});
  a.context.fetch=async url=>url==='/api/frame'?{status:401}:new Promise(resolve=>{finish=resolve;reached()});
  a.elements.get('browser-camera').videoWidth=10;
  vm.runInContext("browserCameraStream={getTracks:()=>[{stop(){}}]};browserFrameCanvas.getContext=()=>({drawImage(){}});browserFrameCanvas.toBlob=cb=>cb({});",a.context);
  const upload=vm.runInContext('uploadBrowserFrames()',a.context);await refreshing;
  a.render({server_instance:'new',token:'new-token',revision:0});
  finish({ok:true,json:async()=>old});await upload;
  assert.equal(vm.runInContext('state.server_instance',a.context),'new');
  assert.equal(vm.runInContext('token',a.context),'new-token');
});

test('new message completion preserves words typed after confirmation', async () => {
  const a=screen();a.render({profile_id:'alice',decision:{text:'Previous',status:'accepted',reason:'corrected'}});
  let finish;
  a.context.fetch=()=>new Promise(resolve=>{finish=resolve});
  const clearing=a.elements.get('new-message').onclick();
  await Promise.resolve();
  a.elements.get('corrected-text').value='Typed while waiting';a.elements.get('corrected-text').listeners.input();
  finish({ok:true,json:async()=>({...a.context.sample,revision:2,decision:{text:'',status:'unavailable',reason:'lip_only'}})});
  await clearing;
  assert.equal(a.elements.get('corrected-text').value,'Typed while waiting');
  assert.equal(a.sessionValues.get('quietbridge-unsent-v1:alice'),'Typed while waiting');
  assert.equal(a.elements.get('speak').disabled,true);
  assert.match(a.elements.get('draft-status').textContent,/newer unfinished edit is kept/);
});

test('hung action times out once, accepts fresh polling and releases the action queue', async () => {
  const a=screen();a.render({server_instance:'boot',token:'token',revision:1,decision:{text:'Old words',status:'accepted',reason:'corrected'}});
  a.elements.get('corrected-text').value='New words';a.elements.get('corrected-text').listeners.input();
  let signal, calls=0;
  a.context.fetch=async(url,options)=>{calls++;signal=options.signal;return new Promise(()=>{})};
  const applying=a.elements.get('save-correction').onclick();await Promise.resolve();
  assert.equal(a.elements.get('save-correction').disabled,true);
  a.context.fetch=async()=>({ok:true,json:async()=>({...a.context.sample,revision:4,decision:{text:'Fresh server words',status:'accepted',reason:'corrected'}})});
  await vm.runInContext('poll()',a.context);
  assert.equal(a.elements.get('message').textContent,'Fresh server words');
  const timeout=[...a.timers.values()].find(t=>t.ms===15000);assert.ok(timeout);timeout.fn();await applying;
  assert.equal(signal.aborted,true);assert.equal(calls,1);
  assert.equal(a.elements.get('corrected-text').value,'New words');
  assert.match(a.elements.get('error').textContent,/result is uncertain/);
  assert.equal(vm.runInContext('pendingActions',a.context),0);
  a.context.fetch=async(url,options)=>({ok:true,json:async()=>({...a.context.sample,revision:5,decision:{text:JSON.parse(options.body).text,status:'accepted',reason:'corrected'}})});
  await a.elements.get('save-correction').onclick();
  assert.equal(a.elements.get('message').textContent,'New words');
});

test('a late older action response cannot clear an edit after a newer result arrives', async () => {
  const a=screen();a.render({server_instance:'boot',token:'token',revision:1,decision:{text:'Previous',status:'accepted',reason:'corrected'}});
  a.elements.get('corrected-text').value='Unfinished draft';a.elements.get('corrected-text').listeners.input();
  let finish;a.context.fetch=()=>new Promise(resolve=>{finish=resolve});
  const clearing=a.elements.get('new-message').onclick();await Promise.resolve();
  a.render({server_instance:'boot',token:'token',revision:3,decision:{text:'Newer incoming words',status:'accepted',reason:'corrected'}});
  finish({ok:true,json:async()=>({...a.context.sample,revision:2,decision:{text:'',status:'unavailable',reason:'lip_only'}})});await clearing;
  assert.equal(a.elements.get('corrected-text').value,'Unfinished draft');
  assert.equal(a.elements.get('message').textContent,'Newer incoming words');
  assert.equal(a.elements.get('speak').disabled,true);
  assert.match(a.elements.get('error').textContent,/newer result/);
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
