const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../ui/story.js'), 'utf8');

function harness({reduced = false, fetch = async () => ({ok:true,json:async()=>({width:28,height:28,glyphs:{}})})} = {}) {
  const classes = new Set(), handlers = {}, timers = [];
  const body = {classList:{add:(...names)=>names.forEach(name=>classes.add(name)),remove:(...names)=>names.forEach(name=>classes.delete(name)),contains:name=>classes.has(name)}};
  const skip = {textContent:'Skip animation',disabled:false,addEventListener:(event,handler)=>{handlers.skip=handler;}};
  const status = {textContent:''};
  let fetchCount = 0;
  const context = {
    document:{body,getElementById:id=>id==='skip-animation'?skip:status,addEventListener:()=>{},querySelectorAll:()=>[]},
    window:{matchMedia:()=>({matches:reduced,addEventListener:()=>{}}),addEventListener:()=>{}},
    fetch:async (...args)=>{fetchCount++;return fetch(...args);},
    setTimeout:(handler,delay)=>{const timer={handler,delay,cancelled:false};timers.push(timer);return timer;},
    clearTimeout:timer=>{if(timer)timer.cancelled=true;},
  };
  vm.runInNewContext(source,context);
  return {classes,skip,status,timers,handlers,get fetchCount(){return fetchCount;}};
}
const flush = () => new Promise(resolve=>setImmediate(resolve));

test('reduced motion keeps readable copy without fetching or waiting', async () => {
  const h=harness({reduced:true});await flush();
  assert.equal(h.fetchCount,0);assert.equal(h.classes.has('is-complete'),true);assert.equal(h.timers.length,0);
});
test('the hand illustration holds for four seconds before the text transition', async () => {
  const h=harness();await flush();
  assert.equal(h.classes.has('is-illustrating'),true);assert.equal(h.classes.has('is-settling'),false);
  assert.equal(h.timers[0].delay,4000);h.timers[0].handler();
  assert.equal(h.classes.has('is-settling'),true);assert.equal(h.timers[1].delay,900);
  h.timers[1].handler();assert.equal(h.classes.has('is-complete'),true);
});
test('skip before data arrives cannot start a delayed animation afterwards', async () => {
  let resolve;
  const pending=new Promise(done=>{resolve=done;});
  const h=harness({fetch:()=>pending});h.handlers.skip();
  resolve({ok:true,json:async()=>({width:28,height:28,glyphs:{}})});await flush();
  assert.equal(h.classes.has('is-complete'),true);assert.equal(h.classes.has('is-illustrating'),false);assert.equal(h.timers.length,0);
});
test('failed illustration data leaves immediately readable content', async () => {
  const h=harness({fetch:async()=>{throw new Error('offline');}});await flush();
  assert.equal(h.classes.has('is-complete'),true);assert.equal(h.classes.has('is-illustrating'),false);
});
