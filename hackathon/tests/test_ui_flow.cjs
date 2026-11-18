const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function screen(page = 'index.html', sessionValues = new Map(), localValues = new Map()) {
  const elements = new Map(), timers = new Map(), windowListeners = new Map(), documentListeners = new Map(), images = [];
  let timerId = 0;
  function element(tag = 'DIV') {
    const classes = new Set();
    return {
      tagName: tag.toUpperCase(), textContent: '', hidden: false, disabled: false,
      value: '', checked: false, style: {}, dataset: {}, children: [], attributes: {},
      classList: { toggle(name, enabled) { if (enabled) classes.add(name); else classes.delete(name); }, add(name) { classes.add(name); }, remove(name) { classes.delete(name); }, contains(name) { return classes.has(name); } },
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
  for (const [markup, tag, id] of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"[^>]*>/g)) { const el = element(tag); el.hidden = /\shidden[\s>]/.test(markup); elements.set(id, el); }
  const context = {
    document: {
      getElementById(id) { return elements.get(id) || null; },
      createElement: element, hidden: false, body: element('BODY'),
      addEventListener(name, fn) { documentListeners.set(name, fn); },
      querySelectorAll(selector) { return selector === '.alternative' ? elements.get('alternatives').children : []; },
    },
    window: { addEventListener(name, fn) { windowListeners.set(name, fn); }, confirm: () => true,
      sessionStorage: {getItem: key => sessionValues.has(key) ? sessionValues.get(key) : null, setItem: (key,value) => sessionValues.set(key,value), removeItem: key => sessionValues.delete(key)},
      localStorage: {getItem: key => localValues.get(key) || null, setItem: (key,value) => localValues.set(key,value)} },
    fetch: () => new Promise(() => {}),
    setTimeout(fn, ms) { const id=++timerId; timers.set(id,{fn,ms});return id; },
    clearTimeout(id) { timers.delete(id); },
    Date, Promise, console, AbortController,
    Image: function () { images.push(this); this.decode = () => new Promise(() => {}); },
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
  return { context, elements, render, sessionValues, localValues, timers, windowListeners, documentListeners, images };
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

// ---- Speech controls, Show large, keyboard apply, polling cadence, local preview ----
const settle = () => new Promise(setImmediate);
function okJson(data) { return { ok: true, status: 200, json: async () => data }; }

test('Stop speaking appears only while the device is speaking and sends one direct request', async () => {
  const a = screen(); a.render({ server_instance: 'boot', token: 'tok', decision: { text: 'Hello', status: 'accepted', reason: 'corrected' } });
  assert.equal(a.elements.get('stop-speaking').hidden, true, 'absent speaking field means not speaking');
  assert.equal(a.elements.get('speak').disabled, false);
  a.render({ server_instance: 'boot', token: 'tok', revision: 2, speaking: true, decision: { text: 'Hello', status: 'accepted', reason: 'corrected' } });
  assert.equal(a.elements.get('stop-speaking').hidden, false);
  assert.equal(a.elements.get('speak').disabled, true, 'no second speech request while speaking');
  // An unrelated action is hung; stopping speech must not wait behind it.
  const calls = [];
  a.context.fetch = async (url, options) => {
    calls.push([url, options.body, options.headers['X-Lipflow-Token']]);
    if (url.endsWith('/stop_speaking')) return okJson({ ...a.context.sample, revision: 3, speaking: false });
    return new Promise(() => {});
  };
  vm.runInContext("requestAction('mode', {mode:'lips'})", a.context);
  await a.elements.get('stop-speaking').onclick(); await settle();
  assert.deepEqual(calls.filter(c => c[0].endsWith('/stop_speaking')), [['/api/action/stop_speaking', '{}', 'tok']]);
  assert.equal(a.elements.get('stop-speaking').hidden, true);
  assert.equal(calls.filter(c => c[0].endsWith('/speak')).length, 0);
});

test('a failed stop-speaking request reports the uncertainty and is not retried', async () => {
  const a = screen(); a.render({ token: 'tok', speaking: true, decision: { text: 'Hello', status: 'accepted', reason: 'corrected' } });
  let calls = 0; a.context.fetch = async () => { calls++; throw new Error('Network down'); };
  await a.elements.get('stop-speaking').onclick();
  assert.equal(calls, 1);
  assert.match(a.elements.get('error').textContent, /still speaking/);
  assert.equal(a.elements.get('stop-speaking').disabled, false);
});

test('Show large shows only applied or explicitly selected text and never speaks', async () => {
  const a = screen(); const calls = []; a.context.fetch = async url => { calls.push(url); return new Promise(() => {}); };
  a.render({ decision: { text: 'I need water', status: 'confirmation_needed', reason: 'conflict' }, alternatives: ['I need water', 'No.'] });
  assert.equal(a.elements.get('show-large').disabled, true, 'an unchosen conflict cannot be shown');
  a.elements.get('show-large').onclick();
  assert.equal(a.elements.get('large-display').hidden, true);
  a.elements.get('alternatives').children[1].onclick();
  assert.equal(a.elements.get('show-large').disabled, false, 'an explicit selection can be shown');
  a.elements.get('corrected-text').value = 'Unapplied words'; a.elements.get('corrected-text').listeners.input();
  assert.equal(a.elements.get('show-large').disabled, true, 'an unapplied edit cannot be shown');
  a.render({ revision: 2, decision: { text: 'Applied words', status: 'accepted', reason: 'corrected' } });
  a.elements.get('corrected-text').value = 'Applied words'; a.elements.get('corrected-text').listeners.input();
  assert.equal(a.elements.get('show-large').disabled, false);
  a.elements.get('show-large').onclick();
  assert.equal(a.elements.get('large-display').hidden, false);
  assert.equal(a.elements.get('large-text').textContent, 'Applied words');
  assert.equal(a.elements.get('large-close').focused, true, 'focus moves into the dialog');
  assert.equal(a.elements.get('main').inert, true, 'background is inert');
  a.elements.get('show-large').focused = false;
  a.documentListeners.get('keydown')({ code: 'Escape', key: 'Escape', target: a.elements.get('large-close'), preventDefault() {} });
  assert.equal(a.elements.get('large-display').hidden, true);
  assert.equal(a.elements.get('main').inert, false);
  assert.equal(a.elements.get('show-large').focused, true, 'focus returns to the opener');
  assert.deepEqual(calls, [], 'showing never requests speech, learning or any action');
});

test('Show large is refused for a recognized but unconfirmed message and during a live session', () => {
  const a = screen();
  a.render({ decision: { text: 'Maybe these words', status: 'confirmation_needed', reason: 'lip_only' } });
  assert.equal(a.elements.get('show-large').disabled, true);
  a.render({ revision: 2, session_active: true, session_text: 'live words', decision: { text: 'live words', status: 'accepted', reason: 'lip_only' } });
  assert.equal(a.elements.get('show-large').disabled, true);
});

test('Escape closes Show large before it cancels a recording', async () => {
  const a = screen(); const calls = []; a.context.fetch = async url => { calls.push(url); return okJson({ ...a.context.sample, revision: 9, phase: 'idle' }); };
  a.render({ phase: 'recording', decision: { text: 'Applied', status: 'accepted', reason: 'corrected' } });
  vm.runInContext('openLargeDisplay()', a.context);
  assert.equal(a.elements.get('large-display').hidden, false);
  const escape = { code: 'Escape', key: 'Escape', target: a.elements.get('large-close'), preventDefault() {} };
  a.documentListeners.get('keydown')(escape); await settle();
  assert.equal(a.elements.get('large-display').hidden, true);
  assert.deepEqual(calls, []);
  a.documentListeners.get('keydown')({ ...escape, target: a.elements.get('show-large') }); await settle();
  assert.deepEqual(calls, ['/api/action/cancel']);
});

test('Ctrl+Enter and Cmd+Enter apply the edit with the button guards and never speak', async () => {
  const a = screen(); a.render({ token: 'tok', decision: { text: 'Old', status: 'accepted', reason: 'corrected' } });
  const calls = []; a.context.fetch = async (url, options) => { calls.push([url, JSON.parse(options.body)]); return okJson({ ...a.context.sample, revision: calls.length + 1, decision: { text: JSON.parse(options.body).text, status: 'accepted', reason: 'corrected' } }); };
  const editor = a.elements.get('corrected-text');
  const key = extra => { let prevented = false; editor.listeners.keydown({ key: 'Enter', preventDefault() { prevented = true; }, ...extra }); return prevented; };
  assert.equal(key({ ctrlKey: true }), true);
  await settle();
  assert.deepEqual(calls, [], 'unchanged text: the disabled button guard applies');
  editor.value = 'New words'; editor.listeners.input();
  assert.equal(key({}), false, 'plain Enter keeps its newline');
  assert.deepEqual(calls, []);
  key({ ctrlKey: true }); await settle(); await vm.runInContext('actionChain', a.context);
  assert.deepEqual(calls, [['/api/action/correct', { revision: 1, text: 'New words', learn: false }]]);
  editor.value = 'Mac words'; editor.listeners.input();
  key({ metaKey: true }); await settle(); await vm.runInContext('actionChain', a.context);
  assert.equal(calls.length, 2);
  assert.equal(calls.some(c => c[0].endsWith('/speak') || c[0].endsWith('/confirm')), false);
});

test('Ctrl+Enter is ignored while another action is pending', async () => {
  const a = screen(); a.render({ decision: { text: 'Old', status: 'accepted', reason: 'corrected' } });
  const editor = a.elements.get('corrected-text'); editor.value = 'Edit'; editor.listeners.input();
  let calls = 0; a.context.fetch = () => { calls++; return new Promise(() => {}); };
  a.elements.get('save-correction').onclick(); await settle();
  editor.listeners.keydown({ key: 'Enter', ctrlKey: true, preventDefault() {} }); await settle();
  assert.equal(calls, 1);
});

test('Space toggling still ignores form controls', async () => {
  const a = screen(); a.render(); let calls = 0; a.context.fetch = () => { calls++; return new Promise(() => {}); };
  for (const tag of ['TEXTAREA', 'INPUT', 'BUTTON']) a.documentListeners.get('keydown')({ code: 'Space', target: { tagName: tag }, preventDefault() {} });
  assert.equal(calls, 0);
  a.documentListeners.get('keydown')({ code: 'Space', target: { tagName: 'MAIN' }, preventDefault() {} }); await settle();
  assert.equal(calls, 1);
});

function pollTimers(a) { const poll = vm.runInContext('poll', a.context); return [...a.timers.values()].filter(t => t.fn === poll); }
test('state polling slows when idle or hidden, never stops, and is fast again on activity', async () => {
  const a = screen('index.html');
  const idle = { rehearsal: true, browser_camera: false, camera: 'rehearsal' };
  a.context.fetch = async url => url === '/api/state' ? okJson({ ...a.context.sample }) : new Promise(() => {});
  a.render(idle);
  vm.runInContext('lastUserActivity = 0', a.context);
  await vm.runInContext('poll()', a.context);
  assert.deepEqual(pollTimers(a).map(t => t.ms), [1000], 'idle polling is slower but scheduled');
  a.context.document.hidden = true;
  await vm.runInContext('poll()', a.context);
  assert.deepEqual(pollTimers(a).map(t => t.ms), [3000], 'hidden tab polls slowly but still polls');
  a.context.document.hidden = false;
  a.documentListeners.get('visibilitychange')();
  assert.deepEqual(pollTimers(a).map(t => t.ms), [0], 'becoming visible polls at once');
  await vm.runInContext('poll()', a.context);
  assert.deepEqual(pollTimers(a).map(t => t.ms), [180], 'recent activity keeps the fast rate');
  vm.runInContext('lastUserActivity = 0', a.context); await vm.runInContext('poll()', a.context);
  a.documentListeners.get('pointerdown')();
  assert.deepEqual(pollTimers(a).map(t => t.ms), [0], 'a user action wakes polling');
});

test('pending actions, sessions and connection loss keep polling live', async () => {
  const a = screen('index.html');
  a.render({ rehearsal: true, browser_camera: false, camera: 'rehearsal' });
  vm.runInContext('lastUserActivity = 0', a.context);
  a.context.document.hidden = true;
  a.context.fetch = async url => url === '/api/state' ? okJson({ ...a.context.sample }) : new Promise(() => {});
  vm.runInContext("requestAction('mode', {mode:'lips'})", a.context);
  await vm.runInContext('poll()', a.context);
  assert.deepEqual(pollTimers(a).map(t => t.ms), [180], 'pending action: fast even when hidden');
  const b = screen('index.html');
  b.context.fetch = async () => { throw new Error('down'); };
  b.render({ rehearsal: true, browser_camera: false, camera: 'rehearsal' });
  vm.runInContext('lastUserActivity = 0', b.context); b.context.document.hidden = true;
  await vm.runInContext('poll()', b.context);
  assert.match(b.elements.get('error').textContent, /Reconnecting/);
  assert.ok(pollTimers(b)[0].ms <= 1000, 'reconnect detection stays at 1 s or faster');
  const c = screen('index.html');
  c.context.fetch = async () => okJson({ ...c.context.sample });
  c.render({ session_active: true, auto_recognition: true });
  vm.runInContext('lastUserActivity = 0', c.context);
  await vm.runInContext('poll()', c.context);
  assert.deepEqual(pollTimers(c).map(t => t.ms), [180], 'active session: fast');
});

test('Settings runs no camera tracking or preview polling', async () => {
  const a = screen('settings.html'); const urls = [];
  a.context.fetch = async url => { urls.push(url); return okJson({ ...a.context.sample }); };
  a.render({ browser_camera: false, rehearsal: false, camera: 'ready' });
  await settle();
  assert.equal(urls.includes('/api/tracking'), false);
  assert.equal(a.images.length, 0);
});

function cameraScreen() {
  const a = screen(); a.render({ server_instance: 'boot', token: 'tok' });
  let stopped = 0; const track = { stop() { stopped++; }, addEventListener() {} };
  const stream = { getTracks: () => [track], getVideoTracks: () => [track] };
  a.context.navigator = { mediaDevices: { getUserMedia: async () => stream } };
  const video = a.elements.get('browser-camera'); video.play = async () => {};
  const urls = [];
  a.context.fetch = async (url, options) => {
    urls.push([url, options?.method || 'GET']);
    if (url === '/api/action/camera_session') { const body = JSON.parse(options.body); return okJson({ ...a.context.sample, revision: a.context.sample.revision + 1, session_active: body.active }); }
    return new Promise(() => {});
  };
  return { a, stream, video, urls, stopped: () => stopped };
}

test('browser camera shows the local mirrored video and never downloads its own frames', async () => {
  const { a, stream, video, urls } = cameraScreen();
  assert.equal(video.hidden, true);
  await a.elements.get('browser-camera-button').onclick();
  assert.equal(video.srcObject, stream);
  assert.equal(video.hidden, false, 'local video is the preview');
  assert.equal(a.elements.get('camera').hidden, true);
  assert.equal(a.elements.get('camera-placeholder').hidden, true);
  assert.equal(a.elements.get('camera-preview').classList.contains('live'), true);
  assert.equal(a.elements.get('preview-status').textContent, 'Camera on · mirrored');
  a.render({ server_instance: 'boot', token: 'tok', revision: 5, camera: 'ready', session_active: true });
  await settle();
  assert.equal(a.images.length, 0, 'no GET /api/frame image polling in browser-camera mode');
  assert.equal(urls.some(([u, m]) => u.startsWith('/api/frame') && m === 'GET'), false);
  assert.equal(urls.some(([u]) => u === '/api/tracking'), true, 'landmark overlay still polls while live');
});

test('stopping the browser camera releases tracks and clears the preview instead of freezing it', async () => {
  const { a, video, stopped } = cameraScreen();
  await a.elements.get('browser-camera-button').onclick();
  await a.elements.get('browser-camera-button').onclick();
  assert.equal(stopped(), 1);
  assert.equal(video.srcObject, null);
  assert.equal(video.hidden, true);
  assert.equal(a.elements.get('camera-placeholder').hidden, false);
  assert.equal(a.elements.get('camera-preview').classList.contains('live'), false);
  assert.equal(a.elements.get('preview-status').textContent, 'Camera off');
  assert.equal(vm.runInContext('trackingWanted()', a.context), false, 'tracking stops with the camera');
});

test('server-owned camera keeps the downloaded preview path', () => {
  const a = screen(); a.render({ browser_camera: false, camera: 'ready' });
  assert.equal(a.images.length, 1);
  assert.match(a.images[0].src, /^\/api\/frame\?t=/);
});

test('frame upload waits for a new video frame and never overlaps', async () => {
  const a = screen(); a.render({ server_instance: 'boot', token: 'tok' });
  const video = a.elements.get('browser-camera'); video.videoWidth = 960; video.currentTime = 1;
  let frameCallback = null; video.requestVideoFrameCallback = cb => { frameCallback = cb; return 7; };
  let uploads = 0, release; a.context.fetch = async (url, options) => { if (url === '/api/frame') { uploads++; return new Promise(r => { release = r; }); } return new Promise(() => {}); };
  vm.runInContext("browserCameraStream={getTracks:()=>[{stop(){}}]};browserFrameCanvas.getContext=()=>({drawImage(){}});browserFrameCanvas.toBlob=cb=>cb({});", a.context);
  const first = vm.runInContext('uploadBrowserFrames(true)', a.context); await settle();
  assert.equal(uploads, 1);
  assert.equal(frameCallback, null, 'nothing scheduled while the upload is in flight');
  release({ ok: true, status: 200 }); await first;
  assert.equal(uploads, 1, 'the next upload waits for a new frame');
  assert.equal(typeof frameCallback, 'function');
  let cb = frameCallback; frameCallback = null; cb(); await settle();
  assert.equal(uploads, 1, 'a frame inside the 40 ms budget is skipped, not encoded');
  assert.equal(typeof frameCallback, 'function', 'and the next frame is awaited');
  vm.runInContext('nextFrameUploadAt = 0', a.context);
  cb = frameCallback; frameCallback = null; cb(); await settle();
  assert.equal(uploads, 2);
  // Fallback without requestVideoFrameCallback: an unchanged frame is not re-encoded.
  delete video.requestVideoFrameCallback; release({ ok: true, status: 200 }); await settle();
  let draws = 0; vm.runInContext('browserFrameCanvas.getContext=()=>({drawImage(){globalThis.__draws=(globalThis.__draws||0)+1}})', a.context);
  await vm.runInContext('uploadBrowserFrames(false)', a.context);
  draws = vm.runInContext('globalThis.__draws||0', a.context);
  assert.equal(draws, 0);
  vm.runInContext('stopBrowserCamera()', a.context);
});
