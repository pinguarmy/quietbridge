const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../ui/video-guide.js'), 'utf8');
function harness() {
  const revoked=[], handlers={}, elements={};let next=0;
  for (const id of ['demo-video-file','demo-video','demo-video-empty','demo-video-status','clear-demo-video']) elements[id]={hidden:false,value:'',textContent:'',files:[],addEventListener:(event,fn)=>{handlers[id+':'+event]=fn;},pause(){},load(){},removeAttribute(name){delete this[name];}};
  vm.runInNewContext(source,{document:{getElementById:id=>elements[id]},window:{addEventListener:(event,fn)=>{handlers[event]=fn;}},URL:{createObjectURL:()=> 'blob:local-'+(++next),revokeObjectURL:url=>revoked.push(url)}});
  const select=file=>{elements['demo-video-file'].files=[file];handlers['demo-video-file:change']();};
  return {elements,handlers,revoked,select};
}
test('local video preview revokes replaced, cleared and unloaded object URLs',()=>{
  const h=harness();h.select({name:'one.mp4',type:'video/mp4'});
  assert.equal(h.elements['demo-video'].src,'blob:local-1');assert.equal(h.elements['demo-video'].hidden,false);
  h.select({name:'two.mp4',type:'video/mp4'});assert.deepEqual(h.revoked,['blob:local-1']);
  h.handlers['clear-demo-video:click']();assert.deepEqual(h.revoked,['blob:local-1','blob:local-2']);assert.equal(h.elements['demo-video'].hidden,true);
  h.select({name:'three.mp4',type:'video/mp4'});h.handlers.pagehide();assert.equal(h.revoked.at(-1),'blob:local-3');
});
test('unsupported files and failed playback return to an honest empty state',()=>{
  const h=harness();h.select({name:'notes.txt',type:'text/plain'});assert.equal(h.elements['demo-video'].hidden,true);assert.match(h.elements['demo-video-status'].textContent,/Choose a video/);
  h.select({name:'bad.mp4',type:'video/mp4'});h.handlers['demo-video:error']();assert.equal(h.elements['demo-video'].hidden,true);assert.deepEqual(h.revoked,['blob:local-1']);assert.match(h.elements['demo-video-status'].textContent,/cannot play/);
});
test('Space on video controls or the guide does not start lip recording',async()=>{
  const handlers={},requests=[];
  const record={disabled:false,addEventListener(){}};
  const context={document:{getElementById:id=>id==='record'?record:null,querySelectorAll:()=>[],addEventListener:(name,fn)=>{handlers[name]=fn;}},window:{addEventListener(){}},setTimeout(){},fetch:url=>{requests.push(url);return new Promise(()=>{});}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../ui/app.js'),'utf8'),context);
  const initial=requests.length;
  for(const target of [{tagName:'VIDEO'},{tagName:'SUMMARY'},{tagName:'DIV',closest:()=>({})}]) handlers.keydown({code:'Space',repeat:false,target,preventDefault(){throw new Error('Video keyboard control was intercepted');}});
  await Promise.resolve();assert.equal(requests.length,initial);
  let prevented=false;handlers.keydown({code:'Space',repeat:false,target:{tagName:'BODY'},preventDefault(){prevented=true;}});
  await Promise.resolve();assert.equal(prevented,true);assert.equal(requests.at(-1),'/api/action/start');
});
