const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

test('AI preferences retain failed entries and warn before losing unsaved changes', async () => {
  const ids=['cleanup-settings-form','cleanup-settings-status','cleanup-settings-error','cleanup-save','cleanup-test','cleanup-base-url','cleanup-model','cleanup-tier','cleanup-enabled','cleanup-api-key'];
  const elements=Object.fromEntries(ids.map(id=>[id,{value:'',checked:false,listeners:{},addEventListener(name,fn){this.listeners[name]=fn;}}]));
  const events={};let fail=true;
  const context={document:{getElementById:id=>elements[id]},window:{addEventListener:(name,fn)=>events[name]=fn},fetch:async url=>({ok:!url.includes('/config')||!fail,json:async()=>url==='/api/state'?{token:'test-token'}:{error:'Invalid model',configured:false,enabled:false}})};
  // Let initial loading complete before exercising actual form event handlers.
  const source=fs.readFileSync(path.join(__dirname,'../ui/cleanup-settings.js'),'utf8').replace('  load();','  globalThis.ready = load();');
  vm.createContext(context);vm.runInContext(source,context);await context.ready;
  elements['cleanup-base-url'].value='https://provider.example/v1';
  elements['cleanup-model'].value='My chosen model';
  elements['cleanup-api-key'].value='test-only-not-a-real-key';
  elements['cleanup-settings-form'].listeners.input();
  let prevented=false;events.beforeunload({preventDefault(){prevented=true;}});assert.equal(prevented,true);
  await elements['cleanup-settings-form'].listeners.submit({preventDefault(){}});
  assert.equal(elements['cleanup-api-key'].value,'test-only-not-a-real-key');
  assert.equal(elements['cleanup-model'].value,'My chosen model');
  assert.match(elements['cleanup-settings-error'].textContent,/Invalid model.*entries are still here/);
  assert.equal(elements['cleanup-save'].disabled,false);
  fail=false;await elements['cleanup-settings-form'].listeners.submit({preventDefault(){}});
  assert.equal(elements['cleanup-api-key'].value,'');
  prevented=false;events.beforeunload({preventDefault(){prevented=true;}});assert.equal(prevented,false);
});
