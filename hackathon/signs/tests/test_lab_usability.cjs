const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '..', 'lab.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

test('rehearsal capture is explicitly disabled by server capability', () => {
  assert.match(script,/captureAvailable=s.capture_available!==false/);
  assert.match(script,/b.disabled=!captureAvailable\|\|working/);
});

test('setup collection preserves edited fields and blocks or confirms discard', () => {
  const source = script.slice(script.indexOf('function collectSetupFields()'), script.indexOf('function renderSetupView()'));
  const values = {emoji:'x', label:'HELLO', display_name:'Edited', concept:'Hi', description:'Wave', intent_template:''};
  const row = {querySelector: selector => ({value: values[selector.match(/"(.*?)"/)[1]]})};
  const elements = {profileDisplayName:{value:'New name'}, gestureTbody:{querySelectorAll:()=>[row]}, promptTbody:{querySelectorAll:()=>[{querySelector:()=>({value:'New phrase'})}]}};
  const profile = {id:'p',name:'Old',gestures:[{...values,display_name:'Original',hint:'keep'}],lip_prompts:['Old']};
  const context = {$:id=>elements[id], activeProfile:profile, currentStage:'setup', setupBaseline:JSON.stringify(profile), confirm:()=>false, renderSetupView:()=>{}};
  vm.createContext(context); vm.runInContext(source,context);
  assert.equal(context.setupDirty(),true);
  assert.equal(context.activeProfile.gestures[0].display_name,'Edited');
  assert.equal(context.activeProfile.gestures[0].hint,'keep');
  assert.equal(context.confirmDiscardSetup(),false);
  assert.equal(context.activeProfile.lip_prompts[0],'New phrase');
  context.confirm=()=>true;
  assert.equal(context.confirmDiscardSetup(),true);
  assert.equal(context.activeProfile.gestures[0].display_name,'Original');
  for (const marker of ["$('btnAddGesture').onclick", "$('btnAddPrompt').onclick"]) {
    assert.match(script.slice(script.indexOf(marker),script.indexOf(marker)+120), /collectSetupFields\(\)/);
  }
});

test('capture validates before camera and releases tracks on save and discard', () => {
  const record = script.slice(script.indexOf("$('record').onclick=async()=>"),script.indexOf("$('discard').onclick=async()=>"));
  for (const check of ["localConsent", "mode==='gestures'", "mode==='lips'", "pairedRequest(currentStage"]) assert.ok(record.indexOf(check)<record.indexOf('await camera()'),check);
  assert.match(record,/recording=false;\s*stopCamera\(\)/);
  const discard = script.slice(script.indexOf("$('discard').onclick=async()=>"),script.indexOf('// Model operations'));
  assert.match(discard,/recording=false;\s*stopCamera\(\)/);
  assert.match(discard,/getTracks\(\).forEach\(t=>t.stop\(\)\)/);
  assert.match(html,/id="localConsent" type="checkbox">/);
  assert.match(html,/id="videoConsent" type="checkbox">/);
});

test('pagehide sends authenticated mode-specific discard using keepalive', () => {
  const handler = script.slice(script.indexOf('function discardOnExit()'),script.indexOf("request('/api/state')"));
  for (const mode of ['gestures','lips','paired']) {
    const calls=[]; let stopped=false;
    const ctx={window:{addEventListener:(_,fn)=>fn()}, recording:true,token:'secret',mode,fetch:(...args)=>{calls.push(args);return Promise.resolve()},stopCamera:()=>{stopped=true}};
    vm.createContext(ctx); vm.runInContext(handler,ctx);
    assert.equal(calls[0][0],`/api/${mode==='gestures'?'discard':mode==='lips'?'mouth/discard':'paired/discard'}`);
    assert.equal(calls[0][1].headers['X-QuietBridge-Token'],'secret');
    assert.equal(calls[0][1].keepalive,true); assert.equal(stopped,true);
  }
});

test('profile dialog names first, traps focus, supports escape and restores focus', () => {
  assert.ok(html.indexOf('id="newProfileName"')<html.indexOf('id="newProfileId"'));
  assert.match(html,/aria-modal="true"/);
  const source=script.slice(script.indexOf('function closeProfileDialog()'),script.indexOf("$('btnConfirmNewProfile').onclick"));
  const elements={newProfileDialog:{hidden:true,addEventListener:(_,fn)=>elements.key=fn,querySelectorAll:()=>[elements.newProfileName,elements.btnCancelNewProfile,elements.btnConfirmNewProfile]},newProfileId:{value:''},newProfileName:{value:''},btnNewProfile:{focus(){focused='opener'}},btnCancelNewProfile:{},btnConfirmNewProfile:{}};
  let focused=''; for(const id of ['newProfileName','btnCancelNewProfile','btnConfirmNewProfile']) Object.assign(elements[id],{focus(){focused=id},getClientRects(){return [1]}});
  const previous={focus(){focused='previous'}}; const document={activeElement:previous};
  const context={$:id=>elements[id],document,confirmDiscardSetup:()=>true,currentStage:'train',activeProfile:{},setupBaseline:null,dialogReturnFocus:null};
  vm.createContext(context); vm.runInContext(source,context);
  elements.btnNewProfile.onclick(); assert.equal(focused,'newProfileName');
  document.activeElement=elements.newProfileName; let prevented=false;
  elements.key({key:'Tab',shiftKey:true,preventDefault(){prevented=true}}); assert.equal(focused,'btnConfirmNewProfile'); assert.equal(prevented,true);
  elements.key({key:'Escape',preventDefault(){}}); assert.equal(elements.newProfileDialog.hidden,true); assert.equal(focused,'opener');
});
