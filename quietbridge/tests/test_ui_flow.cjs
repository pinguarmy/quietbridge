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
      append(...kids) { for (const child of kids) { this.children.push(child); if (child.id) elements.set(child.id, child); } },
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
  assert.match(elements.get('record-hint').textContent, /together/);
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

test('guide starts closed so typing comes first, and opens and closes without changing message state', () => {
  const a=screen(); a.render();
  assert.equal(a.elements.get('first-use-guide').hidden,true,'essentials are one intro line; the editor is not pushed down');
  assert.equal(a.elements.get('open-guide').attributes['aria-expanded'],'false');
  assert.match(fs.readFileSync(path.join(__dirname,'../ui/index.html'),'utf8'),/id="entry-hint">Type a message or use the camera\. Nothing is spoken until you choose Speak\./);
  a.elements.get('open-guide').onclick();
  assert.equal(a.elements.get('first-use-guide').hidden,false);
  assert.equal(a.elements.get('guide-heading').focused,true);
  a.elements.get('guide-dismiss').onclick();
  assert.equal(a.elements.get('first-use-guide').hidden,true);
  assert.equal(a.elements.get('open-guide').focused,true);
  assert.equal(a.elements.get('message').textContent,'Your words will appear here.');
});

test('intro copy stays fixed while session guidance changes inside the camera panel', () => {
  const a=screen(); a.render();
  const intro=a.elements.get('entry-hint').textContent;
  const hints=new Set();
  for (const extra of [{}, {session_active:true,auto_recognition:true}, {session_finalizing:true}, {lips:'warming'}, {sign_input:'motion',session_active:true}]) {
    a.render({revision:a.context.sample.revision+1,...extra}); hints.add(a.elements.get('record-hint').textContent);
    assert.equal(a.elements.get('entry-hint').textContent,intro);
  }
  assert.equal(hints.size,5,'each capture state has its own single instruction');
  assert.doesNotMatch(fs.readFileSync(path.join(__dirname,'../ui/app.js'),'utf8'),/'entry-hint'/,'the intro line is never rewritten at runtime');
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
    assert.equal(a.elements.get('browser-camera-label').textContent,'Start camera');
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
  assert.equal(a.elements.get('browser-camera-label').textContent,'Finish message');
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

// Frame pacing is driven by the fake frame callbacks and timers directly; nothing here reads the wall clock.
function frameScreen() {
  const a = screen(); a.render({ server_instance: 'boot', token: 'tok' });
  const video = a.elements.get('browser-camera'); video.videoWidth = 960; video.currentTime = 1;
  const callbacks = [], cancelled = [], releases = [];
  let nextId = 0;
  video.requestVideoFrameCallback = cb => { cb.id = ++nextId; callbacks.push(cb); return cb.id; };
  video.cancelVideoFrameCallback = id => { cancelled.push(id); const i = callbacks.findIndex(cb => cb.id === id); if (i >= 0) callbacks.splice(i, 1); };
  const counts = { uploads: 0, draws: 0 };
  a.context.fetch = async url => { if (url === '/api/frame') { counts.uploads++; return new Promise(r => { releases.push(r); }); } return new Promise(() => {}); };
  a.context.__countDraw = () => { counts.draws++; };
  vm.runInContext("browserCameraStream={getTracks:()=>[{stop(){}}]};browserFrameCanvas.getContext=()=>({drawImage(){__countDraw()}});browserFrameCanvas.toBlob=cb=>cb({});", a.context);
  // Fires the newest registered frame callback once, as the browser would for one presented frame.
  const frame = async mediaTime => { const cb = callbacks.pop(); assert.equal(typeof cb, 'function', 'a frame callback is registered'); cb(0, { mediaTime }); await settle(); };
  const release = async () => { releases.shift()({ ok: true, status: 200 }); await settle(); };
  return { a, video, callbacks, cancelled, releases, counts, frame, release };
}

test('every new camera frame uploads when idle, with exactly one upload in flight and the newest frame next', async () => {
  const f = frameScreen();
  vm.runInContext('scheduleFrameUpload(); scheduleFrameUpload()', f.a.context);
  assert.equal(f.callbacks.length, 1, 'one watcher per camera generation');
  await f.frame(1.000);
  assert.equal(f.counts.uploads, 1);
  assert.equal(f.callbacks.length, 1, 'frames are still watched while the upload is in flight');
  await f.frame(1.033); await f.frame(1.066);
  vm.runInContext('void uploadBrowserFrames(true)', f.a.context); await settle();
  assert.equal(f.counts.uploads, 1, 'never two uploads in flight, whoever asks');
  assert.equal(f.counts.draws, 1, 'a frame skipped during an upload is not encoded');
  await f.release();
  assert.equal(f.counts.uploads, 2, 'the newest frame is sent as soon as the upload finishes, without queueing the skipped ones');
  assert.equal(f.counts.draws, 2);
  await f.release();
  assert.equal(f.counts.uploads, 2, 'nothing waiting, nothing sent');
  // Each new frame on an idle uploader is sent: no wall-clock budget skips it.
  for (const [i, t] of [1.100, 1.133, 1.166].entries()) { await f.frame(t); assert.equal(f.counts.uploads, 3 + i); await f.release(); }
  await f.frame(1.166);
  assert.equal(f.counts.uploads, 5, 'an unchanged frame (same capture time) is not re-encoded');
  await f.frame(1.176);
  assert.equal(f.counts.uploads, 5, 'a camera faster than 30 fps sends at most ~30 frames per second of capture time');
  await f.frame(1.200);
  assert.equal(f.counts.uploads, 6);
  assert.equal(f.counts.draws, 6);
  // Stop cancels the watcher, and the in-flight completion starts nothing.
  await f.frame(1.233);
  const stale = f.callbacks.at(-1);
  vm.runInContext('stopBrowserCamera()', f.a.context);
  assert.equal(f.cancelled.length, 1);
  assert.equal(f.callbacks.length, 0, 'stop cancels the watcher');
  await f.release();
  assert.equal(f.counts.uploads, 6, 'no upload after the camera stopped');
  stale(0, { mediaTime: 2 }); await settle();
  assert.equal(f.counts.uploads, 6, 'a late frame callback of a stopped camera does nothing');
  assert.equal(f.callbacks.length, 0);
});

test('a camera restarted during an upload owns one loop; the old upload cannot start a second one', async () => {
  const f = frameScreen();
  vm.runInContext('scheduleFrameUpload()', f.a.context);
  await f.frame(1.0);
  assert.equal(f.counts.uploads, 1);
  vm.runInContext("stopBrowserCamera(); browserCameraStream={getTracks:()=>[{stop(){}}]}; scheduleFrameUpload()", f.a.context);
  assert.equal(f.callbacks.length, 1);
  await f.frame(0.5);
  assert.equal(f.counts.uploads, 2, 'the new camera uploads its first frame at once');
  await f.release();
  assert.equal(f.callbacks.length, 1, 'the old completion registers no extra watcher');
  await f.frame(0.533);
  assert.equal(f.counts.uploads, 2, 'the new camera still has its own upload in flight');
  await f.release();
  assert.equal(f.counts.uploads, 3, 'its waiting frame follows');
  vm.runInContext('stopBrowserCamera()', f.a.context);
});

test('without requestVideoFrameCallback a 30 fps timer uploads only changed frames and stop cancels it', async () => {
  const f = frameScreen();
  delete f.video.requestVideoFrameCallback;
  vm.runInContext('scheduleFrameUpload(); scheduleFrameUpload()', f.a.context);
  const timers = () => [...f.a.timers.entries()].filter(([, t]) => t.ms === 33);
  assert.equal(timers().length, 1, 'one timer at the 30 fps target');
  const tick = async () => { const [[id, t]] = timers(); f.a.timers.delete(id); t.fn(); await settle(); };
  await tick();
  assert.equal(f.counts.uploads, 1);
  await tick();
  assert.equal(f.counts.uploads, 1, 'no overlap');
  await f.release();
  assert.equal(f.counts.draws, 1, 'the waiting check found the same frame and did not re-encode it');
  await tick();
  assert.equal(f.counts.draws, 1, 'an unchanged frame is never re-encoded');
  f.video.currentTime = 1.04; await tick();
  assert.equal(f.counts.uploads, 2);
  await f.release();
  assert.equal(timers().length, 1, 'still exactly one timer');
  vm.runInContext('stopBrowserCamera()', f.a.context);
  assert.equal(timers().length, 0, 'stop cancels the timer');
});

test('frame uploads stay off in rehearsal', async () => {
  const f = frameScreen();
  f.a.render({ server_instance: 'boot', token: 'tok', rehearsal: true, revision: 2 });
  await vm.runInContext('uploadBrowserFrames(true)', f.a.context);
  assert.equal(f.counts.uploads, 0); assert.equal(f.counts.draws, 0);
});

// ---- Capture warning and live camera rate ----
test('a capture warning is advice beside a settled result: shown, cleared, never blocking Speak', () => {
  const a = screen();
  const result = { decision: { text: 'I need water', status: 'accepted', reason: 'lip_only' } };
  a.render(result);
  assert.equal(a.elements.get('capture-warning').textContent, '', 'absent field shows nothing');
  const warning = 'The camera sent too few pictures, so some words may be missing. Add light or close other camera apps, then record again.';
  a.render({ ...result, revision: 2, capture_warning: warning });
  assert.equal(a.elements.get('capture-warning').textContent, warning);
  assert.equal(a.elements.get('speak').disabled, false, 'Speak stays available');
  assert.equal(a.elements.get('error').textContent, '', 'not an error');
  a.render({ ...result, revision: 2, capture_warning: warning, session_active: true, session_text: 'I need' });
  assert.equal(a.elements.get('capture-warning').textContent, '', 'hidden while live words arrive');
  a.render({ ...result, revision: 3, capture_warning: warning });
  assert.equal(a.elements.get('capture-warning').textContent, warning);
  a.render({ ...result, revision: 4, capture_warning: '' });
  assert.equal(a.elements.get('capture-warning').textContent, '', 'cleared when the field clears');
  a.render({ ...result, revision: 5, capture_warning: 7 });
  assert.equal(a.elements.get('capture-warning').textContent, '', 'a malformed field shows nothing');
  const html = fs.readFileSync(path.join(__dirname, '../ui/index.html'), 'utf8');
  assert.match(html, /<p id="capture-warning" class="callout callout-warn capture-warning" role="status">/, 'a polite status');
  assert.ok(html.indexOf('id="message-actions"') < html.indexOf('id="capture-warning"'), 'below the Speak controls, so it never moves them');
});

test('the live camera rate badge appears only while the camera is live and warns with hysteresis', () => {
  const a = screen();
  let now = 1_000_000;
  a.context.Date = class extends Date { static now() { return now; } };
  const badge = a.elements.get('camera-rate');
  const show = (fps, advance = 200) => { now += advance; a.render({ metrics: fps === undefined ? {} : { camera_input_fps: fps } }); };
  show(29.5);
  assert.equal(badge.hidden, true, 'camera off: nothing to show');
  vm.runInContext("browserCameraStream={getTracks:()=>[{stop(){}}]}", a.context);
  show(12, 0);
  assert.equal(badge.hidden, true, 'a fresh camera is not judged on the previous reading');
  show(29.5, 1500);
  assert.equal(badge.hidden, false);
  assert.equal(badge.textContent, '30 fps');
  assert.equal(badge.classList.contains('badge-warn'), false);
  show(undefined);
  assert.equal(badge.hidden, true, 'absent field: nothing to show');
  show(17);
  for (let i = 0; i < 9; i++) show(17);
  assert.equal(badge.classList.contains('badge-warn'), false, 'a dip shorter than 2 s does not warn');
  show(25); show(17);
  for (let i = 0; i < 9; i++) show(17);
  assert.equal(badge.classList.contains('badge-warn'), false, 'a recovery restarts the hold');
  show(17);
  assert.equal(badge.classList.contains('badge-warn'), true, 'below 20 fps for 2 s warns');
  assert.equal(badge.classList.contains('badge-plain'), false);
  assert.equal(badge.textContent, '17 fps · low', 'not colour alone');
  assert.match(a.elements.get('camera-rate-note').textContent, /light/);
  for (let i = 0; i < 20; i++) show(21);
  assert.equal(badge.classList.contains('badge-warn'), true, '20–22 fps does not clear the warning (no flicker)');
  for (let i = 0; i < 10; i++) show(23);
  assert.equal(badge.classList.contains('badge-warn'), true, 'recovery also needs 2 s');
  show(23);
  assert.equal(badge.classList.contains('badge-warn'), false);
  assert.equal(badge.textContent, '23 fps');
  assert.equal(a.elements.get('camera-rate-note').textContent, '');
  for (let i = 0; i < 11; i++) show(15);
  assert.equal(badge.classList.contains('badge-warn'), true);
  vm.runInContext('stopBrowserCamera()', a.context);
  assert.equal(badge.hidden, true, 'hidden when the camera stops');
  assert.equal(badge.classList.contains('badge-warn'), false);
  assert.equal(a.elements.get('camera-rate-note').textContent, '');
});

// ---- Quick phrases, newer-model notice, speech failure, long/RTL text, profile list ----
const phraseButtons = a => a.elements.get('quick-phrases-list').children;
function phraseScreen(extra = {}) {
  const a = screen(); a.render({ server_instance: 'boot', token: 'tok', profile_id: 'sam', profile_name: 'Sam', quick_phrases: ['I need a break, please.', 'Thank you.'], ...extra });
  return a;
}

test('quick phrases: none for the stock configuration or older state, a labelled group of full-text buttons otherwise', () => {
  const a = screen(); a.render();
  assert.equal(a.elements.get('quick-phrases').hidden, true, 'absent field shows nothing');
  a.render({ revision: 2, quick_phrases: [] });
  assert.equal(a.elements.get('quick-phrases').hidden, true, 'no built-in phrases');
  const long = 'Could you please repeat that more slowly? '.repeat(20).trim();
  a.render({ revision: 3, quick_phrases: ['Yes.', long, '', 7] });
  assert.equal(a.elements.get('quick-phrases').hidden, false);
  const buttons = phraseButtons(a);
  assert.deepEqual(buttons.map(b => b.textContent), ['Yes.', long], 'full text is the accessible name; invalid entries are skipped');
  assert.equal(buttons[1].title, long);
  assert.equal(buttons[1].attributes.dir, 'auto');
  assert.equal(buttons[0].type, 'button');
});

test('a quick phrase applies through the typed-message path without speaking or learning', async () => {
  const a = phraseScreen();
  a.elements.get('learn-correction').checked = true;
  const calls = [];
  a.context.fetch = async (url, options) => { calls.push([url, JSON.parse(options.body)]); return okJson({ ...a.context.sample, revision: 2, decision: { text: 'Thank you.', status: 'accepted', reason: 'corrected' } }); };
  await phraseButtons(a)[1].onclick(); await settle();
  assert.deepEqual(calls, [['/api/action/correct', { revision: 1, text: 'Thank you.', learn: false }]]);
  assert.equal(a.elements.get('message').textContent, 'Thank you.');
  assert.equal(a.elements.get('corrected-text').value, 'Thank you.');
  assert.equal(a.elements.get('learn-correction').checked, false);
  assert.equal(a.elements.get('speak').disabled, false, 'Speak stays a separate explicit action');
  assert.equal(a.elements.get('speak').focused, true);
  assert.equal(phraseButtons(a)[1].disabled, true, 'the current message cannot be re-applied');
  assert.equal(phraseButtons(a)[1].attributes['aria-current'], 'true');
  assert.equal(phraseButtons(a)[0].disabled, false);
  await phraseButtons(a)[1].onclick(); await settle();
  assert.equal(calls.length, 1);
});

test('a quick phrase asks before replacing an unfinished edit and keeps it on decline or failure', async () => {
  const a = phraseScreen({ decision: { text: 'Old', status: 'accepted', reason: 'corrected' } });
  a.elements.get('corrected-text').value = 'My unfinished words'; a.elements.get('corrected-text').listeners.input();
  let asked = 0; a.context.window.confirm = () => { asked++; return false; };
  let calls = 0; a.context.fetch = async () => { calls++; throw new Error('Network unavailable'); };
  await phraseButtons(a)[0].onclick();
  assert.equal(asked, 1); assert.equal(calls, 0);
  assert.equal(a.elements.get('corrected-text').value, 'My unfinished words');
  a.context.window.confirm = () => true;
  await phraseButtons(a)[0].onclick(); await settle();
  assert.equal(calls, 1, 'one request, never retried');
  assert.equal(a.elements.get('corrected-text').value, 'My unfinished words', 'a failed apply keeps the edit');
  assert.equal(a.sessionValues.get('quietbridge-unsent-v1:sam'), 'My unfinished words');
  assert.match(a.elements.get('error').textContent, /edit is kept/);
});

test('words typed while a phrase is applying are not cleared by its success', async () => {
  const a = phraseScreen();
  let finish; a.context.fetch = () => new Promise(resolve => { finish = resolve; });
  const applying = phraseButtons(a)[0].onclick(); await settle();
  assert.equal(phraseButtons(a)[1].disabled, true, 'phrases wait for the pending action');
  assert.equal(a.elements.get('save-correction').disabled, true);
  a.elements.get('corrected-text').value = 'Typed meanwhile'; a.elements.get('corrected-text').listeners.input();
  finish(okJson({ ...a.context.sample, revision: 2, decision: { text: 'I need a break, please.', status: 'accepted', reason: 'corrected' } }));
  await applying;
  assert.equal(a.elements.get('corrected-text').value, 'Typed meanwhile');
  assert.equal(a.elements.get('speak').disabled, true, 'the newer edit is unapplied, so Speak waits');
  assert.match(a.elements.get('draft-status').textContent, /newer unfinished edit is kept/);
});

test('quick phrases are unavailable during sessions, speech and connection loss', async () => {
  const a = phraseScreen(); const P = { quick_phrases: ['I need a break, please.', 'Thank you.'] };
  const blocked = () => phraseButtons(a).length === 2 && phraseButtons(a).every(b => b.disabled);
  assert.equal(blocked(), false);
  a.render({ ...P, revision: 2, session_active: true, auto_recognition: true }); assert.equal(blocked(), true, 'live session');
  a.render({ ...P, revision: 3, session_finalizing: true }); assert.equal(blocked(), true, 'finalizing');
  a.render({ ...P, revision: 4, speaking: true }); assert.equal(blocked(), true, 'device speaking');
  a.render({ ...P, revision: 5 }); assert.equal(blocked(), false);
  a.context.fetch = async () => { throw new Error('down'); };
  await vm.runInContext('poll()', a.context);
  assert.equal(blocked(), true, 'connection lost');
  let calls = 0; a.context.fetch = async () => { calls++; return new Promise(() => {}); };
  await phraseButtons(a)[0].onclick(); assert.equal(calls, 0);
});

const visibleChips = list => list.children.filter(b => !b.hidden);
test('a long phrase list starts collapsed and expands on request', () => {
  const a = screen(); const phrases = Array.from({ length: 30 }, (_, i) => `Phrase ${i + 1}`);
  a.render({ quick_phrases: phrases });
  assert.equal(visibleChips(a.elements.get('quick-phrases-list')).length, 12, 'without measurable layout: a fixed count');
  const more = a.elements.get('quick-phrases-more');
  assert.equal(more.hidden, false); assert.match(more.textContent, /Show all 30/); assert.equal(more.attributes['aria-expanded'], 'false');
  more.onclick();
  assert.equal(visibleChips(a.elements.get('quick-phrases-list')).length, 30); assert.equal(more.attributes['aria-expanded'], 'true');
  a.render({ revision: 2, quick_phrases: ['Only one'] });
  assert.equal(phraseButtons(a).length, 1); assert.equal(more.hidden, true);
});

// Fake layout: chips flow into rows of `perRow` (a 390 px phone fits one long phrase per row).
function withRows(list, perRow) {
  list.children.forEach((b, i) => { Object.defineProperty(b, 'offsetTop', { configurable: true, get: () => (b.hidden ? 0 : Math.floor(i / perRow) * 52) }); });
}
test('phones: collapsed phrases show only what fits in three rows, the same rule at every width', () => {
  const a = screen();
  const long = Array.from({ length: 10 }, (_, i) => `Could you please help me with number ${i + 1}, thank you so much`);
  a.render({ quick_phrases: long });
  const list = a.elements.get('quick-phrases-list'), more = a.elements.get('quick-phrases-more');
  withRows(list, 1); vm.runInContext('layoutPhraseLists()', a.context);
  assert.equal(visibleChips(list).length, 3, 'one long phrase per row on a phone: three rows');
  assert.equal(more.hidden, false); assert.equal(more.textContent, 'Show all 10 phrases');
  assert.equal(list.children.slice(3).every(b => b.hidden), true, 'hidden chips leave the tab order');
  withRows(list, 4); a.windowListeners.get('resize')(); [...a.timers.values()].filter(t => t.ms === 50).forEach(t => t.fn());
  assert.equal(visibleChips(list).length, 10, 'a wide screen fits all ten in three rows');
  assert.equal(more.hidden, true);
  withRows(list, 1); vm.runInContext('layoutPhraseLists()', a.context);
  more.onclick();
  assert.equal(visibleChips(list).length, 10); assert.equal(more.textContent, 'Show fewer phrases');
  a.render({ revision: 2, quick_phrases: long });
  assert.equal(visibleChips(list).length, 10, 'polling does not collapse an expanded list');
});

// ---- First-run setup (state contract: setup.assets / reload_needed) ----
const LICENCE = 'Lip-reading weights: non-commercial research use only.';
function setupState(assets, extra = {}) {
  return { server_instance: 'boot', token: 'tok', profile_id: 'sam', setup: { busy: false, reload_needed: false, assets, ...extra } };
}
const lipsAsset = extra => ({ id: 'lips', label: 'Lip reading', status: 'missing', bytes_done: 0, bytes_total: 0, error: '', size_note: 'about 1.2 GB', license_note: LICENCE, ...extra });

test('setup: absent or all-ready setup renders nothing and requests nothing', async () => {
  for (const page of ['index.html', 'settings.html']) {
    const a = screen(page); const calls = []; a.context.fetch = async url => { calls.push(url); return new Promise(() => {}); };
    a.render();
    assert.equal(a.elements.get('setup-assets').children.length, 0, page);
    if (page === 'index.html') assert.equal(a.elements.get('setup-panel').hidden, true);
    else assert.equal(a.elements.get('downloads').hidden, true);
    a.render({ ...setupState([lipsAsset({ status: 'ready' })]), revision: 2 });
    if (page === 'index.html') assert.equal(a.elements.get('setup-panel').hidden, true, 'nothing to do on the message page');
    else assert.equal(a.elements.get('downloads').hidden, false, 'Settings lists what is installed');
    await settle();
    assert.deepEqual(calls.filter(u => u.startsWith('/api/setup')), []);
  }
});

test('setup: a missing asset explains itself, shows the licence before an unticked consent, and Download needs the consent', async () => {
  const a = screen(); const calls = [];
  a.context.fetch = async (url, options) => { calls.push([url, JSON.parse(options.body), options.headers['X-Lipflow-Token']]); return okJson({ ok: true }); };
  a.render(setupState([lipsAsset()]));
  assert.equal(a.elements.get('setup-panel').hidden, false);
  assert.match(a.elements.get('setup-explain-lips').textContent, /needs a download \(about 1\.2 GB\)\./);
  assert.match(a.elements.get('setup-intro').textContent, /Typing works now.*Nothing downloads until you choose/);
  assert.equal(a.elements.get('setup-license-lips').hidden, false);
  assert.match(a.elements.get('setup-license-lips').textContent, /non-commercial research use only/);
  const item = a.elements.get('setup-assets').children[0].children;
  const order = cls => item.findIndex(c => c.className === cls);
  assert.ok(order('setup-license') < order('check setup-consent') && order('check setup-consent') < order('setup-actions'), 'licence, then consent, then Download');
  const consent = a.elements.get('setup-consent-lips');
  assert.equal(consent.checked, false, 'consent starts unticked');
  assert.equal(consent.type, 'checkbox');
  assert.match(a.elements.get('setup-progress-lips').attributes['aria-labelledby'], /setup-title-lips/);
  await a.elements.get('setup-download-lips').onclick();
  assert.deepEqual(calls, [], 'no consent: nothing is requested');
  assert.equal(consent.focused, true, 'focus moves to the consent box');
  assert.match(a.elements.get('setup-hint-lips').textContent, /tick the box/i);
  assert.match(a.elements.get('record-hint').textContent, /one-time download/);
  assert.equal(a.elements.get('corrected-text').disabled, false, 'typing stays available');
  consent.checked = true; consent.listeners.change();
  assert.equal(a.elements.get('setup-hint-lips').textContent, '');
  await a.elements.get('setup-download-lips').onclick();
  assert.deepEqual(calls, [['/api/setup/download', { id: 'lips', accept_license: true }, 'tok']]);
});

test('setup: download progress, polite sparse announcements, cancel, error and Try again', async () => {
  const a = screen(); const calls = [];
  a.context.fetch = async (url, options) => { calls.push([url, JSON.parse(options.body)]); return okJson({ ok: true }); };
  a.render(setupState([lipsAsset()]));
  assert.equal(a.elements.get('setup-announce').textContent, '', 'the first render announces nothing');
  a.render({ ...setupState([lipsAsset({ status: 'downloading', bytes_done: 340e6, bytes_total: 1.2e9 })], { busy: true }), revision: 2 });
  assert.equal(a.elements.get('setup-progress-lips').max, 1.2e9);
  assert.equal(a.elements.get('setup-progress-lips').value, 340e6);
  assert.equal(a.elements.get('setup-bytes-lips').textContent, '340 MB of 1.20 GB · 28%');
  assert.equal(a.elements.get('setup-download-lips').hidden, true);
  assert.doesNotMatch(a.elements.get('setup-intro').textContent, /until you choose/, 'no offer wording while it downloads');
  assert.equal(a.elements.get('setup-assets').children[0].children.find(c => c.className === 'check setup-consent').hidden, true, 'no consent box while downloading');
  assert.equal(a.elements.get('setup-cancel-lips').hidden, false);
  assert.equal(a.elements.get('setup-announce').textContent, 'Lip reading download started.');
  a.elements.get('setup-announce').textContent = '';
  a.render({ ...setupState([lipsAsset({ status: 'downloading', bytes_done: 350e6, bytes_total: 1.2e9 })], { busy: true }), revision: 3 });
  assert.equal(a.elements.get('setup-announce').textContent, '', 'not on every poll');
  assert.equal(a.elements.get('setup-bytes-lips').textContent, '350 MB of 1.20 GB · 29%');
  a.render({ ...setupState([lipsAsset({ status: 'downloading', bytes_done: 600e6, bytes_total: 1.2e9 })], { busy: true }), revision: 4 });
  assert.equal(a.elements.get('setup-announce').textContent, 'Lip reading download 50% done.', 'each quarter is announced once');
  const removed = []; a.elements.get('setup-progress-lips').removeAttribute = name => removed.push(name);
  a.render({ ...setupState([lipsAsset({ status: 'downloading', bytes_done: 5e6, bytes_total: 0 })], { busy: true }), revision: 5 });
  assert.deepEqual(removed, ['value'], 'unknown size: an indeterminate bar');
  assert.equal(a.elements.get('setup-bytes-lips').textContent, '5 MB so far', 'unknown size: indeterminate, bytes only');
  await a.elements.get('setup-cancel-lips').onclick();
  assert.deepEqual(calls, [['/api/setup/cancel', { id: 'lips' }]]);
  a.render({ ...setupState([lipsAsset({ status: 'missing', bytes_done: 5e6 })]), revision: 6 });
  assert.match(a.elements.get('setup-announce').textContent, /cancelled/);
  a.render({ ...setupState([lipsAsset({ status: 'error', error: 'Network unreachable.' })]), revision: 7 });
  assert.equal(a.elements.get('setup-error-lips').hidden, false);
  assert.equal(a.elements.get('setup-error-lips').textContent, 'Network unreachable.');
  assert.equal(a.elements.get('setup-error-lips').attributes.role, 'alert');
  assert.equal(a.elements.get('setup-download-lips').textContent, 'Try again');
  assert.equal(a.elements.get('setup-license-lips').hidden, false, 'licence shown again before retrying');
  a.elements.get('setup-consent-lips').checked = false;
  await a.elements.get('setup-download-lips').onclick();
  assert.equal(calls.length, 1, 'Try again also needs the consent');
  a.elements.get('setup-consent-lips').checked = true;
  await a.elements.get('setup-download-lips').onclick();
  assert.deepEqual(calls[1], ['/api/setup/download', { id: 'lips', accept_license: true }]);
});

test('setup: a refused or hung download request is reported once and never retried', async () => {
  const a = screen(); let calls = 0;
  a.context.fetch = async () => { calls++; return { ok: false, status: 400, json: async () => ({ error: 'Accept the licence first.' }) }; };
  a.render(setupState([lipsAsset()]));
  a.elements.get('setup-consent-lips').checked = true;
  await a.elements.get('setup-download-lips').onclick(); await settle();
  assert.equal(calls, 1);
  assert.match(a.elements.get('setup-hint-lips').textContent, /could not start: Accept the licence first/);
  a.render({ ...setupState([lipsAsset()]), revision: 2 });
  assert.match(a.elements.get('setup-hint-lips').textContent, /could not start/, 'the failure stays visible across polls');
  a.context.fetch = () => { calls++; return new Promise(() => {}); };
  const pending = a.elements.get('setup-download-lips').onclick(); await settle();
  assert.equal(a.elements.get('setup-download-lips').disabled, true);
  assert.equal(a.elements.get('setup-download-lips').textContent, 'Starting…');
  [...a.timers.values()].find(t => t.ms === 15000).fn(); await pending;
  assert.equal(calls, 2);
  assert.match(a.elements.get('setup-hint-lips').textContent, /timed out.*uncertain/);
  assert.equal(a.elements.get('setup-download-lips').disabled, false);
});

test('setup: while one download runs, another cannot start', () => {
  const a = screen();
  a.render(setupState([lipsAsset({ status: 'downloading', bytes_done: 1, bytes_total: 10 }), { id: 'gestures', label: 'Gestures', status: 'missing', license_note: 'Apache-2.0', size_note: 'about 8 MB' }], { busy: true }));
  assert.equal(a.elements.get('setup-download-gestures').disabled, true);
  assert.match(a.elements.get('setup-hint-gestures').textContent, /Another download is running/);
});

test('setup: Ready explains the reload, is blocked while the camera is in use, and reloads only on request', async () => {
  const a = screen(); const calls = [];
  a.context.fetch = async (url, options) => { calls.push([url, options?.body && JSON.parse(options.body)]); return okJson({ ...a.context.sample, revision: 9, lips: 'warming', setup: { busy: false, reload_needed: false, assets: [lipsAsset({ status: 'ready' })] } }); };
  a.render({ ...setupState([lipsAsset({ status: 'ready' })], { reload_needed: true }), session_active: true, auto_recognition: true });
  assert.equal(a.elements.get('setup-panel').hidden, false);
  assert.equal(a.elements.get('setup-ready').hidden, false);
  assert.equal(a.elements.get('setup-reload').disabled, true);
  assert.match(a.elements.get('setup-reload-blocked').textContent, /stop the camera/);
  assert.match(fs.readFileSync(path.join(__dirname, '../ui/index.html'), 'utf8'), /id="setup-ready-detail">[^<]*clears the current message/);
  await a.elements.get('setup-reload').onclick();
  await settle();
  assert.deepEqual(calls.filter(c => c[0] !== '/api/state' && c[0] !== '/api/tracking'), [], 'state alone and a blocked button never reload');
  a.render({ ...setupState([lipsAsset({ status: 'ready' })], { reload_needed: true }), revision: 2, decision: { text: 'Current words', status: 'accepted', reason: 'corrected' } });
  a.elements.get('corrected-text').value = 'Typed edit'; a.elements.get('corrected-text').listeners.input();
  assert.equal(a.elements.get('setup-reload').disabled, false);
  let asked = 0; a.context.window.confirm = message => { asked++; assert.match(message, /clears the current message.*edit stays/); return false; };
  await a.elements.get('setup-reload').onclick();
  assert.equal(asked, 1); assert.equal(calls.filter(c => c[0].startsWith('/api/action')).length, 0);
  a.context.window.confirm = () => true;
  await a.elements.get('setup-reload').onclick();
  assert.deepEqual(calls.filter(c => c[0].startsWith('/api/action')), [['/api/action/profile_select', { id: 'sam' }]]);
  assert.equal(a.elements.get('setup-panel').hidden, true, 'done: the panel goes away');
  assert.equal(a.elements.get('corrected-text').value, 'Typed edit');
  assert.match(a.elements.get('correction-status').textContent, /Loading the new models/);
});

test('setup: Settings shows the same downloads compactly, including installed ones', async () => {
  const a = screen('settings.html'); const calls = [];
  a.context.fetch = async (url, options) => { calls.push([url, JSON.parse(options.body)]); return okJson({ ok: true }); };
  a.render(setupState([lipsAsset({ status: 'ready' }), { id: 'gestures', label: 'Gestures', status: 'missing', license_note: 'Apache-2.0', size_note: 'about 8 MB' }]));
  assert.equal(a.elements.get('downloads').hidden, false);
  assert.equal(a.elements.get('setup-assets').children.length, 2);
  assert.match(a.elements.get('setup-explain-lips').textContent, /Downloaded/);
  assert.equal(a.elements.get('setup-download-lips').hidden, true);
  await a.elements.get('setup-download-gestures').onclick();
  assert.deepEqual(calls, []);
  a.elements.get('setup-consent-gestures').checked = true;
  await a.elements.get('setup-download-gestures').onclick();
  assert.deepEqual(calls, [['/api/setup/download', { id: 'gestures', accept_license: true }]]);
});

test('setup: polling re-renders in place so the consent tick and focus survive', () => {
  const a = screen(); a.render(setupState([lipsAsset()]));
  const before = a.elements.get('setup-assets').children[0];
  a.elements.get('setup-consent-lips').checked = true;
  let replaced = 0; const original = a.elements.get('setup-assets').replaceChildren;
  a.elements.get('setup-assets').replaceChildren = function (...c) { replaced++; return original.apply(this, c); };
  for (let r = 2; r < 6; r++) a.render({ ...setupState([lipsAsset()]), revision: r });
  assert.equal(replaced, 0);
  assert.equal(a.elements.get('setup-assets').children[0], before);
  assert.equal(a.elements.get('setup-consent-lips').checked, true);
});

// ---- Voice settings (Settings) ----
const voiceState = extra => ({ server_instance: 'boot', token: 'tok', speech_settings: { rate: 'normal', voice: '', voices: [{ id: 'en-gb', label: 'English (UK)' }, { id: 'en-us', label: 'English (US)' }] }, ...extra });
test('voice: absent settings show nothing; rate and voice save explicitly with visible feedback and never speak', async () => {
  const a = screen('settings.html'); a.render();
  assert.equal(a.elements.get('voice').hidden, true, 'absent field: no Voice card');
  const calls = [];
  a.context.fetch = async (url, options) => { const body = JSON.parse(options.body); calls.push([url, body, options.headers['X-Lipflow-Token']]); return okJson({ ...a.context.sample, revision: a.context.sample.revision + 1, speech_settings: { ...a.context.sample.speech_settings, ...body } }); };
  a.render({ ...voiceState(), revision: 2 });
  assert.equal(a.elements.get('voice').hidden, false);
  assert.equal(a.elements.get('voice-rate-normal').attributes['aria-pressed'], 'true');
  assert.deepEqual(a.elements.get('voice-select').children.map(o => [o.value, o.textContent]), [['', 'System default'], ['en-gb', 'English (UK)'], ['en-us', 'English (US)']]);
  await a.elements.get('voice-rate-slow').onclick();
  assert.deepEqual(calls, [['/api/speech/settings', { rate: 'slow' }, 'tok']]);
  assert.equal(a.elements.get('voice-rate-slow').attributes['aria-pressed'], 'true');
  assert.match(a.elements.get('voice-status').textContent, /Saved/);
  await a.elements.get('voice-rate-slow').onclick();
  assert.equal(calls.length, 1, 'choosing the current rate sends nothing');
  a.elements.get('voice-select').value = 'en-us'; await a.elements.get('voice-select').listeners.change();
  assert.deepEqual(calls[1], ['/api/speech/settings', { voice: 'en-us' }, 'tok']);
  assert.equal(calls.some(c => /speak/.test(c[0].replace('/api/speech/settings', ''))), false);
});

test('voice: Test voice asks for one sample, never touches the message and is not retried', async () => {
  const a = screen('settings.html');
  const calls = [];
  let fail = false;
  a.context.fetch = async (url, options) => { calls.push([url, JSON.parse(options.body)]); return fail ? { ok: false, status: 400, json: async () => ({ error: 'Speech is already playing. Stop it before testing the voice' }) } : okJson({ ...a.context.sample, revision: a.context.sample.revision }); };
  a.render({ ...voiceState(), revision: 2 });
  await a.elements.get('voice-test').onclick();
  assert.deepEqual(calls, [['/api/speech/test', {}]]);
  assert.match(a.elements.get('voice-status').textContent, /Your message was not changed/);
  assert.equal(a.elements.get('voice-test').disabled, false);
  fail = true;
  await a.elements.get('voice-test').onclick();
  assert.equal(calls.length, 2, 'a refused sample is reported, not retried');
  assert.match(a.elements.get('voice-error').textContent, /not spoken/);
  a.render({ ...voiceState(), revision: 3, speaking: true });
  assert.equal(a.elements.get('voice-test').disabled, true, 'unavailable while a message is being spoken');
});

test('voice: a failed save says so, is not retried and returns to the saved voice', async () => {
  const a = screen('settings.html'); a.render(voiceState());
  let calls = 0; a.context.fetch = async () => { calls++; return { ok: false, status: 400, json: async () => ({ error: 'Unknown voice' }) }; };
  a.elements.get('voice-select').value = 'en-gb'; await a.elements.get('voice-select').listeners.change(); await settle();
  assert.equal(calls, 1);
  assert.match(a.elements.get('voice-error').textContent, /Not saved: Unknown voice/);
  assert.equal(a.elements.get('voice-status').textContent, '');
  assert.equal(a.elements.get('voice-select').value, '', 'shows the voice actually in use');
  a.render({ ...voiceState(), revision: 2 });
  assert.equal(calls, 1);
});

test('voice: no voices or no speech hides the choices with a short explanation', () => {
  const a = screen('settings.html');
  a.render(voiceState({ speech_settings: { rate: 'fast', voice: '', voices: [] } }));
  assert.equal(a.elements.get('voice-select-field').hidden, true);
  assert.match(a.elements.get('voice-list-note').textContent, /only its standard voice/);
  assert.equal(a.elements.get('voice-controls').hidden, false);
  a.render(voiceState({ revision: 2, speech_available: false }));
  assert.equal(a.elements.get('voice-controls').hidden, true);
  assert.match(a.elements.get('voice-unavailable').textContent, /not available/);
  // Test voice uses its own backend route (a fixed sample that leaves the message alone) and lives
  // inside the controls, so it disappears with them when this computer has no speech.
  const html = fs.readFileSync(path.join(__dirname, '../ui/settings.html'), 'utf8');
  const controls = html.slice(html.indexOf('id="voice-controls"'), html.indexOf('id="voice-unavailable"'));
  assert.match(controls, /id="voice-test"/);
});

test('voice: polling does not undo a choice in progress, but follows a real change', () => {
  const a = screen('settings.html'); a.render(voiceState());
  const select = a.elements.get('voice-select');
  select.value = 'en-gb';
  a.render({ ...voiceState(), revision: 2 });
  assert.equal(select.value, 'en-gb');
  a.render({ ...voiceState({ speech_settings: { rate: 'normal', voice: 'en-us', voices: voiceState().speech_settings.voices } }), revision: 3 });
  assert.equal(select.value, 'en-us');
});

// ---- About line and Quit ----
test('about line comes from app and is absent for older state', () => {
  const a = screen('settings.html'); a.render();
  assert.equal(a.elements.get('about-line').textContent, '');
  assert.equal(a.elements.get('quit-card').hidden, true, 'an older backend is not offered Quit');
  a.render({ revision: 2, app: { version: '0.9.0', started_with: 'browser_camera' } });
  assert.equal(a.elements.get('about-line').textContent, 'QuietBridge 0.9.0 · started using this browser’s camera');
  a.render({ revision: 3, app: { version: '0.9.0', started_with: 'rehearsal' } });
  assert.match(a.elements.get('about-line').textContent, /preview mode, without a camera/);
  assert.equal(a.elements.get('quit-card').hidden, false);
});

function quitScreen(sessionValues = new Map()) {
  const a = screen('settings.html', sessionValues);
  a.render({ server_instance: 'boot-a', token: 'tok', profile_id: 'sam', app: { version: '1', started_with: 'browser_camera' } });
  return a;
}
test('quit asks first, stops on one request, shows a calm stopped page with the draft, and polls without reconnect noise', async () => {
  const storage = new Map([['quietbridge-unsent-v1:sam', 'Half-typed words']]);
  const a = quitScreen(storage); const calls = [];
  a.context.window.confirm = () => false;
  a.context.fetch = async (url, options) => { calls.push([url, options?.body]); return okJson({ ok: true }); };
  await a.elements.get('quit-button').onclick();
  assert.deepEqual(calls, [], 'declined: nothing sent');
  a.context.window.confirm = message => { assert.match(message, /stops the camera and QuietBridge on this computer/); return true; };
  await a.elements.get('quit-button').onclick();
  assert.deepEqual(calls, [['/api/shutdown', '{}']]);
  assert.equal(a.elements.get('main').hidden, true);
  assert.equal(a.elements.get('stopped-screen').hidden, false);
  assert.equal(a.elements.get('stopped-heading').focused, true);
  assert.equal(a.elements.get('stopped-draft-block').hidden, false);
  assert.equal(a.elements.get('stopped-draft').value, 'Half-typed words', 'the unfinished edit stays visible and copyable');
  assert.equal(a.context.document.title, 'QuietBridge has stopped');
  a.context.fetch = async () => { throw new Error('connection refused'); };
  await vm.runInContext('poll()', a.context);
  assert.equal(a.elements.get('error').textContent, '', 'no "Connection lost… reconnecting" after a deliberate quit');
  assert.deepEqual(pollTimers(a).map(t => t.ms), [3000], 'quiet, slow checks for a restart');
  assert.equal(storage.get('quietbridge-unsent-v1:sam'), 'Half-typed words');
  a.context.fetch = async () => okJson({ ...a.context.sample, server_instance: 'boot-a', revision: 5 });
  await vm.runInContext('poll()', a.context);
  assert.equal(a.elements.get('stopped-screen').hidden, false, 'the app that was quit answering once more is not a restart');
  a.context.fetch = async () => okJson({ ...a.context.sample, server_instance: 'boot-b', token: 'tok-b', revision: 0 });
  await vm.runInContext('poll()', a.context);
  assert.equal(a.elements.get('stopped-screen').hidden, true, 'a restarted app is picked up as today');
  assert.equal(a.elements.get('main').hidden, false);
  assert.match(a.elements.get('error').textContent, /restarted/);
  assert.equal(vm.runInContext('token', a.context), 'tok-b');
});

test('a failed quit says it may still be running and is not retried', async () => {
  const a = quitScreen(); let calls = 0;
  a.context.fetch = async () => { calls++; throw new Error('Network down'); };
  await a.elements.get('quit-button').onclick(); await settle();
  assert.equal(calls, 1);
  assert.match(a.elements.get('quit-error').textContent, /may still be running/);
  assert.equal(a.elements.get('stopped-screen').hidden, true);
  assert.equal(a.elements.get('quit-button').disabled, false);
});

test('quit with no unfinished edit hides the draft block', async () => {
  const a = quitScreen();
  a.context.fetch = async () => okJson({ ok: true });
  await a.elements.get('quit-button').onclick();
  assert.equal(a.elements.get('stopped-draft-block').hidden, true);
});

// ---- Recent messages (this tab only) ----
function recentScreen(storage = new Map(), extra = {}) {
  const a = screen('index.html', storage);
  a.render({ server_instance: 'boot', token: 'tok', profile_id: 'sam', ...extra });
  return a;
}
const recentList = a => a.elements.get('recent-list').children.map(b => b.textContent);
let recentRevision = 10;
async function applyAndSpeak(a, words, speechError = '') {
  a.context.fetch = async (url, options) => {
    const body = JSON.parse(options.body);
    const rev = ++recentRevision;
    if (url.endsWith('/correct')) return okJson({ ...a.context.sample, revision: rev, decision: { text: body.text, status: 'accepted', reason: 'corrected' } });
    if (url.endsWith('/speak')) return okJson({ ...a.context.sample, revision: rev, last_spoken: body.text, speech_error: speechError });
    return new Promise(() => {});
  };
  a.elements.get('corrected-text').value = words; a.elements.get('corrected-text').listeners.input();
  await a.elements.get('save-correction').onclick();
  await a.elements.get('speak').onclick();
}
test('recent: only messages actually dispatched without a speech error, or shown large, are kept', async () => {
  const a = recentScreen();
  assert.equal(a.elements.get('recent-card').hidden, true, 'nothing yet');
  a.elements.get('corrected-text').value = 'Applied only'; a.elements.get('corrected-text').listeners.input();
  a.context.fetch = async (url, options) => okJson({ ...a.context.sample, revision: 2, decision: { text: 'Applied only', status: 'accepted', reason: 'corrected' } });
  await a.elements.get('save-correction').onclick();
  assert.equal(a.elements.get('recent-card').hidden, true, 'applying is not speaking');
  await applyAndSpeak(a, 'Speech failed', 'No audio device');
  assert.equal(a.elements.get('recent-card').hidden, true, 'a speech error is not recorded');
  await applyAndSpeak(a, 'Please open the window');
  assert.deepEqual(recentList(a), ['Please open the window']);
  assert.equal(a.elements.get('recent-card').hidden, false);
  a.render({ revision: 500, server_instance: 'boot', token: 'tok', profile_id: 'sam', decision: { text: 'Shown large words', status: 'accepted', reason: 'corrected' } });
  a.elements.get('show-large').onclick();
  assert.deepEqual(recentList(a), ['Shown large words', 'Please open the window'], 'newest first');
  assert.equal(JSON.parse(a.sessionValues.get('quietbridge-recent-v1:sam')).length, 2, 'stored only in this tab, per profile');
});

test('recent: de-duplicated, capped at eight, per profile, and clearing removes only this list', async () => {
  const storage = new Map([['quietbridge-unsent-v1:sam', 'Draft stays']]);
  const a = recentScreen(storage);
  for (let i = 1; i <= 10; i++) vm.runInContext(`rememberRecent(${JSON.stringify('Message ' + i)})`, a.context);
  vm.runInContext("rememberRecent('Message 5')", a.context);
  assert.equal(recentList(a).length, 8);
  assert.deepEqual(recentList(a).slice(0, 3), ['Message 5', 'Message 10', 'Message 9']);
  const other = recentScreen(storage, { profile_id: 'alex' });
  assert.equal(other.elements.get('recent-card').hidden, true, 'another profile has its own list');
  storage.set('quietbridge-recent-v1:alex', '{not json');
  const broken = recentScreen(storage, { profile_id: 'alex' });
  assert.equal(broken.elements.get('recent-card').hidden, true, 'unreadable storage is ignored');
  a.elements.get('recent-clear').onclick();
  assert.equal(a.elements.get('recent-card').hidden, true);
  assert.equal(storage.has('quietbridge-recent-v1:sam'), false);
  assert.equal(storage.get('quietbridge-unsent-v1:sam'), 'Draft stays');
  assert.equal(a.elements.get('corrected-text').focused, true);
});

test('recent: storage that throws never breaks typing', () => {
  const throwing = { has: () => { throw new Error('denied'); }, get: () => { throw new Error('denied'); }, set: () => { throw new Error('denied'); }, delete: () => { throw new Error('denied'); } };
  const a = recentScreen(throwing);
  vm.runInContext("rememberRecent('Hello')", a.context);
  assert.equal(a.elements.get('recent-card').hidden, true);
  assert.equal(a.elements.get('corrected-text').disabled, false);
});

test('recent: choosing one sets the message like a phrase and never speaks', async () => {
  const a = recentScreen(new Map([['quietbridge-recent-v1:sam', JSON.stringify(['Thank you.', 'Water, please.'])]]));
  assert.deepEqual(recentList(a), ['Thank you.', 'Water, please.']);
  const calls = [];
  a.context.fetch = async (url, options) => { calls.push([url, JSON.parse(options.body)]); return okJson({ ...a.context.sample, revision: 2, decision: { text: 'Water, please.', status: 'accepted', reason: 'corrected' } }); };
  await a.elements.get('recent-list').children[1].onclick(); await settle();
  assert.deepEqual(calls, [['/api/action/correct', { revision: 1, text: 'Water, please.', learn: false }]]);
  assert.equal(a.elements.get('recent-list').children[1].disabled, true, 'the current message');
  assert.equal(a.elements.get('recent-list').children[1].attributes['aria-current'], 'true');
  a.render({ revision: 3, server_instance: 'boot', token: 'tok', profile_id: 'sam', session_active: true, auto_recognition: true });
  assert.equal(a.elements.get('recent-list').children.every(b => b.disabled), true, 'unavailable during a live session');
});

// ---- "Your phrase" suggestions ----
test('phrase matches are labelled "Your phrase" in the option name and are never preselected', () => {
  const a = screen();
  const decision = { text: 'I need a brake', status: 'accepted', reason: 'lip_only' };
  a.render({ decision, alternatives: ['I need a brake', 'I need a break, please.'], phrase_matches: ['I need a break, please.'] });
  const buttons = a.elements.get('alternatives').children;
  assert.equal(buttons.length, 2, 'an accepted result with an own-phrase match still offers the choice');
  assert.equal(buttons[0].children.length, 0); assert.equal(buttons[0].textContent, 'I need a brake');
  const name = buttons[1].children.map(c => c.textContent).join('');
  assert.equal(name, 'Your phrase: I need a break, please.', 'the accessible name includes the label');
  assert.equal(buttons[1].children[1].className, 'sr-only');
  assert.equal(buttons[1].attributes['aria-pressed'], 'false', 'never preselected');
  assert.equal(buttons[0].attributes['aria-pressed'], 'true', 'the recognised words stay the current choice');
  assert.equal(a.elements.get('message').textContent, 'I need a brake');
  buttons[1].onclick();
  assert.equal(a.elements.get('message').textContent, 'I need a break, please.');
  assert.equal(a.elements.get('speak-label').textContent, 'Confirm & speak');
  a.render({ revision: 2, decision, alternatives: ['I need a brake', 'I need a break, please.'] });
  assert.equal(a.elements.get('alternatives').children.length, 0, 'absent field: no extra choices for an accepted result');
});

test('phrase matches during a conflict keep the conflict rules', () => {
  const a = screen();
  a.render({ decision: { text: 'Yes', status: 'confirmation_needed', reason: 'conflict' }, alternatives: ['Yes', 'No.', 'Not now, thank you.'], phrase_matches: ['Not now, thank you.', 'Not in alternatives'] });
  const buttons = a.elements.get('alternatives').children;
  assert.equal(buttons.length, 3);
  assert.equal(buttons.every(b => b.attributes['aria-pressed'] === 'false'), true);
  assert.equal(a.elements.get('speak').disabled, true, 'a conflict still needs a choice');
  assert.equal(buttons[2].classList.contains('own-phrase'), true);
});

test('newer model notice explains the consequence, is blocked while the camera is in use and never reloads by itself', async () => {
  const a = screen(); const calls = [];
  a.context.fetch = async (url, options) => { calls.push([url, options?.body && JSON.parse(options.body)]); return new Promise(() => {}); };
  a.render({ server_instance: 'boot', token: 'tok', profile_id: 'sam' });
  assert.equal(a.elements.get('models-reload-notice').hidden, true, 'absent field: no notice');
  a.render({ server_instance: 'boot', token: 'tok', profile_id: 'sam', revision: 2, models_reload_needed: true, models_reload_detail: 'Promoted gesture model differs.', session_active: true, auto_recognition: true });
  assert.equal(a.elements.get('models-reload-notice').hidden, false);
  assert.equal(a.elements.get('models-reload-detail').textContent, 'Promoted gesture model differs.');
  assert.equal(a.elements.get('models-reload-button').disabled, true);
  assert.match(a.elements.get('models-reload-blocked').textContent, /stop the camera/);
  await a.elements.get('models-reload-button').onclick();
  a.render({ server_instance: 'boot', token: 'tok', profile_id: 'sam', revision: 3, models_reload_needed: true, models_reload_detail: 'Promoted gesture model differs.', phase: 'recording' });
  assert.equal(a.elements.get('models-reload-button').disabled, true);
  a.render({ server_instance: 'boot', token: 'tok', profile_id: 'sam', revision: 4, models_reload_needed: true });
  assert.equal(a.elements.get('models-reload-button').disabled, false);
  assert.equal(a.elements.get('models-reload-blocked').textContent, '');
  assert.match(a.elements.get('models-reload-detail').textContent, /differs from the one running/);
  assert.deepEqual(calls.filter(c => c[0].startsWith('/api/action') || c[0].startsWith('/api/profile')), [], 'state alone never triggers a reload');
});

test('reloading from the notice confirms when a message would be cleared, keeps the typed edit and hides on fresh state', async () => {
  const a = screen();
  a.render({ server_instance: 'boot', token: 'tok', profile_id: 'sam', models_reload_needed: true, models_reload_detail: 'New', decision: { text: 'Current words', status: 'accepted', reason: 'corrected' } });
  a.elements.get('corrected-text').value = 'Typed edit'; a.elements.get('corrected-text').listeners.input();
  const calls = []; let asked = 0;
  a.context.window.confirm = message => { asked++; assert.match(message, /clears the current message.*edit stays/); return false; };
  a.context.fetch = async (url, options) => { calls.push([url, JSON.parse(options.body)]); return okJson({ ...a.context.sample, revision: 2, models_reload_needed: false, decision: { text: '', status: 'unavailable', reason: 'lip_only' } }); };
  await a.elements.get('models-reload-button').onclick();
  assert.equal(asked, 1); assert.deepEqual(calls, []);
  a.context.window.confirm = () => true;
  await a.elements.get('models-reload-button').onclick();
  assert.deepEqual(calls, [['/api/action/profile_select', { id: 'sam' }]]);
  assert.equal(a.elements.get('models-reload-notice').hidden, true);
  assert.equal(a.elements.get('corrected-text').value, 'Typed edit');
  assert.match(a.elements.get('correction-status').textContent, /reloaded/);
});

test('Settings reflects a newer model next to Reload current models and reloads the running profile', async () => {
  const a = screen('settings.html');
  a.render({ token: 'tok', profile_id: 'sam', models_reload_needed: true, models_reload_detail: 'Lip model candidate 2 was promoted.' });
  assert.equal(a.elements.get('models-reload-notice').hidden, false);
  assert.equal(a.elements.get('models-reload-detail').textContent, 'Lip model candidate 2 was promoted.');
  assert.equal(a.elements.get('profile-reload-btn').classList.contains('reload-needed'), true);
  a.render({ token: 'tok', profile_id: 'sam', revision: 2, session_active: true, models_reload_needed: true });
  const calls = []; a.context.fetch = async (url, options) => { calls.push(url); return okJson({ ...a.context.sample, revision: 3 }); };
  await a.elements.get('profile-reload-btn').onclick();
  assert.deepEqual(calls, [], 'refused while a session is active');
  assert.match(a.elements.get('profile-switch-error').textContent, /stop the camera/);
  a.render({ token: 'tok', profile_id: 'sam', revision: 4 });
  assert.equal(a.elements.get('models-reload-notice').hidden, true);
  assert.equal(a.elements.get('profile-reload-btn').classList.contains('reload-needed'), false);
});

test('a speech failure is shown at Speak with fallbacks and is not presented as spoken', () => {
  const a = screen();
  const accepted = { text: 'Open the window', status: 'accepted', reason: 'corrected' };
  a.render({ decision: accepted, last_spoken: 'Open the window' });
  assert.equal(a.elements.get('step-speak').attributes['aria-current'], 'step');
  assert.equal(a.elements.get('speech-error').textContent, '');
  a.render({ revision: 2, decision: accepted, last_spoken: 'Open the window', speech_error: 'No audio output device was found' });
  assert.match(a.elements.get('speech-error').textContent, /Speech did not play: No audio output device was found\. Use Show large or Copy message/);
  assert.equal(a.elements.get('last-spoken').textContent, '', 'not listed as spoken');
  assert.equal(a.elements.get('step-speak').attributes['aria-current'], undefined);
  assert.equal(a.elements.get('step-review').attributes['aria-current'], 'step');
  assert.equal(a.elements.get('show-large').classList.contains('show-large-primary'), true);
  assert.equal(a.elements.get('speak').disabled, false, 'trying again stays an explicit choice');
  a.render({ revision: 3, decision: accepted, last_spoken: 'Open the window', speech_error: '' });
  assert.equal(a.elements.get('speech-error').textContent, '');
  assert.match(a.elements.get('last-spoken').textContent, /Last spoken/);
});

test('long and right-to-left messages stay readable', () => {
  const a = screen();
  const long = 'word '.repeat(400).trim();
  a.render({ decision: { text: long, status: 'accepted', reason: 'corrected' } });
  assert.equal(a.elements.get('message').classList.contains('long'), true);
  assert.equal(a.elements.get('message').attributes.tabindex, '0', 'scrollable message is keyboard reachable');
  a.render({ revision: 2, decision: { text: 'أحتاج إلى استراحة', status: 'confirmation_needed', reason: 'conflict' }, alternatives: ['أحتاج إلى استراحة', 'No.'] });
  assert.equal(a.elements.get('message').classList.contains('long'), false);
  assert.equal(a.elements.get('message').attributes.tabindex, undefined);
  assert.equal(a.elements.get('alternatives').children[0].attributes.dir, 'auto');
  const html = fs.readFileSync(path.join(__dirname, '../ui/index.html'), 'utf8');
  for (const id of ['message', 'corrected-text', 'large-text']) assert.match(html, new RegExp(`id="${id}"[^>]*dir="auto"`));
});

test('a profile chosen in the list survives polling until the running profile changes', () => {
  const { elements, context, render } = screen('settings.html');
  vm.runInContext('profilesLoaded = true', context);
  render({ profile_id: 'alice', profile_name: 'Alice' });
  assert.equal(elements.get('profile-select').value, 'alice');
  elements.get('profile-select').value = 'bob';  // chosen, focus already moved on
  render({ profile_id: 'alice', profile_name: 'Alice', revision: 2 });
  render({ profile_id: 'alice', profile_name: 'Alice', revision: 3 });
  assert.equal(elements.get('profile-select').value, 'bob', 'a poll must not undo the choice before Use profile');
  render({ profile_id: 'carol', profile_name: 'Carol', revision: 4 });
  assert.equal(elements.get('profile-select').value, 'carol', 'a real profile change is still shown');
});

test('a failed or empty profile list is explained, disables switching and retries after polling recovers', async () => {
  const a = screen('settings.html'); let calls = 0;
  a.context.fetch = async url => { if (url === '/api/profiles') { calls++; return { ok: false, json: async () => ({}) }; } return okJson({ ...a.context.sample }); };
  a.render({ token: 'tok', profile_id: 'default' });
  await vm.runInContext('profilesLoading = false; loadProfilesList()', a.context);
  assert.equal(a.elements.get('profile-switch-btn').disabled, true);
  assert.match(a.elements.get('profile-switch-error').textContent, /Trying again automatically/);
  assert.equal(a.elements.get('profile-select').children[0].textContent, 'Profiles unavailable');
  await vm.runInContext('poll()', a.context); await settle();
  assert.equal(calls, 1, 'retry waits a few seconds');
  vm.runInContext('profilesRetryAt = 0', a.context);
  a.context.fetch = async url => url === '/api/profiles' ? okJson({ profiles: [], active_profile: 'default' }) : okJson({ ...a.context.sample });
  await vm.runInContext('poll()', a.context); await settle();
  assert.equal(a.elements.get('profile-select').children[0].textContent, 'No saved profiles yet');
  assert.equal(a.elements.get('profile-switch-error').textContent, '');
  assert.match(a.elements.get('profile-list-status').textContent, /Personalize/);
  assert.equal(a.elements.get('profile-reload-btn').disabled, false, 'reloading the running profile does not need the list');
});

test('no page loads the removed cleanup-display script', () => {
  const ui = path.join(__dirname, '../ui');
  assert.equal(fs.existsSync(path.join(ui, 'cleanup-display.js')), false);
  for (const page of ['index.html', 'settings.html']) assert.doesNotMatch(fs.readFileSync(path.join(ui, page), 'utf8'), /cleanup-display/);
});

test('without on-device speech the unusable Speak button is hidden and Show large leads', () => {
  const a = screen(); a.render({ speech_available: false, decision: { text: 'Hello', status: 'accepted', reason: 'corrected' } });
  assert.equal(a.elements.get('speak').hidden, true);
  assert.equal(a.elements.get('speak').disabled, true);
  assert.equal(a.elements.get('show-large').classList.contains('show-large-primary'), true);
  assert.equal(a.elements.get('show-large').disabled, false);
  a.render({ revision: 2, speech_available: true, decision: { text: 'Hello', status: 'accepted', reason: 'corrected' } });
  assert.equal(a.elements.get('speak').hidden, false);
  assert.equal(a.elements.get('show-large').classList.contains('show-large-primary'), false);
});

// ---- Presentation contracts of the redesigned page (class toggles only; no behaviour change) ----
test('Use this message takes the primary style only while an applicable edit is unapplied', () => {
  const a = screen(); a.render({ decision: { text: 'Old words', status: 'accepted', reason: 'corrected' } });
  const save = a.elements.get('save-correction');
  assert.equal(save.classList.contains('btn-primary'), false, 'nothing to apply: quiet');
  a.elements.get('corrected-text').value = 'New words'; a.elements.get('corrected-text').listeners.input();
  assert.equal(save.disabled, false);
  assert.equal(save.classList.contains('btn-primary'), true, 'unapplied edit: applying is the next step');
  assert.equal(a.elements.get('speak').disabled, true, 'and Speak waits meanwhile');
  assert.equal(a.elements.get('decision-status').classList.contains('badge-warn'), true);
  a.elements.get('corrected-text').value = ''; a.elements.get('corrected-text').listeners.input();
  assert.equal(save.disabled, true);
  assert.equal(save.classList.contains('btn-primary'), false, 'a disabled button is never styled as the primary action');
  a.elements.get('corrected-text').value = 'Old words'; a.elements.get('corrected-text').listeners.input();
  assert.equal(save.classList.contains('btn-primary'), false);
});

test('exactly one hero action: Speak, Stop speaking while speaking, or Show large without reliable speech', () => {
  const a = screen(); const accepted = { text: 'Hello', status: 'accepted', reason: 'corrected' };
  const actions = a.elements.get('message-actions'), speak = a.elements.get('speak'), show = a.elements.get('show-large');
  a.render({ decision: accepted });
  assert.equal(speak.classList.contains('btn-hero'), true); assert.equal(show.classList.contains('btn-hero'), false);
  assert.equal(actions.classList.contains('speaking'), false);
  a.render({ revision: 2, decision: accepted, speaking: true });
  assert.equal(actions.classList.contains('speaking'), true, 'Stop speaking takes the hero slot');
  assert.equal(a.elements.get('stop-speaking').hidden, false);
  a.render({ revision: 3, decision: accepted, speech_error: 'No audio output device was found' });
  assert.equal(actions.classList.contains('show-leads'), true);
  assert.equal(show.classList.contains('btn-hero'), true); assert.equal(speak.classList.contains('btn-hero'), false);
  assert.equal(speak.hidden, false, 'Speak stays available as a deliberate retry');
  a.render({ revision: 4, decision: accepted, speech_available: false });
  assert.equal(show.classList.contains('btn-primary'), true); assert.equal(speak.hidden, true);
});

test('the recording toggle uses the danger style only while recording', () => {
  const a = screen(); a.render({ sign_input: 'motion', session_active: true, auto_recognition: true });
  const record = a.elements.get('record');
  assert.equal(record.classList.contains('btn-danger'), false);
  a.render({ revision: 2, sign_input: 'motion', session_active: true, auto_recognition: true, phase: 'recording' });
  assert.equal(record.classList.contains('btn-danger'), true);
  assert.equal(a.elements.get('record-label').textContent, 'Finish recording');
  assert.equal(record.attributes['aria-pressed'], 'true');
});

test('the camera button label lives in its own span so the icon survives every label change', () => {
  const html = fs.readFileSync(path.join(__dirname, '../ui/index.html'), 'utf8');
  assert.match(html, /id="browser-camera-button"[^>]*>(?:<svg[\s\S]*?<\/svg>)+<span id="browser-camera-label">Start camera<\/span><\/button>/);
  assert.match(html, /id="speak"[^>]*>(?:<svg[\s\S]*?<\/svg>)<span id="speak-label">/);
  assert.doesNotMatch(fs.readFileSync(path.join(__dirname, '../ui/app.js'), 'utf8'), /\$\('browser-camera-button'\)[^;]*textContent|button\.textContent = 'Start camera'/);
});

test('pages honour the strict CSP: no inline style, script or data URLs, and both load the design system first', () => {
  for (const page of ['index.html', 'settings.html']) {
    const html = fs.readFileSync(path.join(__dirname, '../ui/', page), 'utf8');
    assert.doesNotMatch(html, /\sstyle=|<style|<script(?![^>]*\bsrc=)|data:|blob:|(?:src|href)="https?:/, page);
    assert.match(html, /<link rel="stylesheet" href="\/design\.css"><link rel="stylesheet" href="\/style\.css">/, page);
    assert.match(html, /<link rel="icon" href="\/favicon\.svg" type="image\/svg\+xml">/, page);
    assert.equal((html.match(/<h1[\s>]/g) || []).length, 1, `${page} has exactly one h1`);
    for (const svg of html.match(/<svg[^>]*>/g) || []) assert.match(svg, /aria-hidden="true" focusable="false"/, `${page}: decorative icon hidden`);
  }
  assert.doesNotMatch(fs.readFileSync(path.join(__dirname, '../ui/app.js'), 'utf8'), /\.style\./, 'JS toggles classes, never element.style');
});
