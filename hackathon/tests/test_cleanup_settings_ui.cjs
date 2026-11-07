const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function settingsScreen(fetch) {
  const ids=['cleanup-settings-form','cleanup-settings-status','cleanup-settings-error','cleanup-save','cleanup-test','cleanup-base-url','cleanup-model','cleanup-tier','cleanup-enabled','cleanup-api-key'];
  const elements=Object.fromEntries(ids.map(id=>[id,{value:'',checked:false,listeners:{},addEventListener(name,fn){this.listeners[name]=fn;}}]));
  const events={};
  const context={document:{getElementById:id=>elements[id]},window:{addEventListener:(name,fn)=>events[name]=fn},fetch};
  vm.createContext(context);
  // Execute the shipped script unchanged. Drive the same form events as the browser.
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../ui/cleanup-settings.js'),'utf8'),context);
  return {elements,events,settle:()=>new Promise(setImmediate)};
}
const ok = data => ({ok:true,json:async()=>data});
function edit(a, values) {
  for (const [id,value] of Object.entries(values)) a.elements['cleanup-'+id].value=value;
  a.elements['cleanup-settings-form'].listeners.input();
}
function warns(a) {
  let prevented=false;a.events.beforeunload({preventDefault(){prevented=true}});return prevented;
}

test('AI preferences retain failed entries and warn before losing unsaved changes', async () => {
  let fail=true;
  const a=settingsScreen(async(url,options)=>!options?ok(url==='/api/state'?{token:'test-token'}:{}):{ok:!fail,json:async()=>({error:'Invalid model',configured:false})});
  await a.settle();
  edit(a, {'base-url':'https://provider.example/v1',model:'My chosen model','api-key':'test-only-not-a-real-key'});
  assert.equal(warns(a),true);
  await a.elements['cleanup-settings-form'].listeners.submit({preventDefault(){}});
  assert.equal(a.elements['cleanup-api-key'].value,'test-only-not-a-real-key');
  assert.equal(a.elements['cleanup-model'].value,'My chosen model');
  assert.match(a.elements['cleanup-settings-error'].textContent,/Invalid model.*entries are still here/);
  assert.equal(a.elements['cleanup-save'].disabled,false);
  fail=false;await a.elements['cleanup-settings-form'].listeners.submit({preventDefault(){}});
  assert.equal(a.elements['cleanup-api-key'].value,'');
  assert.equal(warns(a),false);
});

test('a slow preferences load never overwrites entries typed while waiting', async () => {
  let finish;
  const response=new Promise(resolve=>{finish=resolve});
  const a=settingsScreen(async url=>url==='/api/state'?ok({token:'local-token'}):response);
  await a.settle();
  edit(a,{model:'New user input'});
  finish(ok({model:'Old saved model',enabled:false}));await a.settle();
  assert.equal(a.elements['cleanup-model'].value,'New user input');
  assert.equal(warns(a),true);
});

test('saving older preferences never clears newer entries or the unsaved warning', async () => {
  let finish;
  const a=settingsScreen(async(url,options)=>options?new Promise(resolve=>{finish=resolve}):ok(url==='/api/state'?{token:'local-token'}:{model:'Initial model',enabled:false}));
  await a.settle();
  edit(a,{model:'Model sent','api-key':'synthetic-first'});
  const saving=a.elements['cleanup-settings-form'].listeners.submit({preventDefault(){}});
  edit(a,{model:'Newer model','api-key':'synthetic-newer'});
  finish(ok({configured:true}));await saving;
  assert.equal(a.elements['cleanup-api-key'].value,'synthetic-newer');
  assert.equal(a.elements['cleanup-model'].value,'Newer model');
  assert.equal(warns(a),true);
  assert.match(a.elements['cleanup-settings-status'].textContent,/unsaved/i);
});
