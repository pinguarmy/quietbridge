const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../ui/story.js'), 'utf8');

const chartData = () => ({kind:'bsl-chart',width:744,height:1215,image:'/assets/bsl-alphabet.png',glyphs:{A:{crop:[12,9,103,100]}}});
function harness({reduced = false, fetch = async () => ({ok:true,json:async()=>chartData()})} = {}) {
  const classes = new Set(), handlers = {}, timers = [];
  const body = {classList:{add:(...names)=>names.forEach(name=>classes.add(name)),remove:(...names)=>names.forEach(name=>classes.delete(name)),contains:name=>classes.has(name)}};
  const skip = {textContent:'Skip animation',disabled:false,hidden:true,addEventListener:(event,handler)=>{handlers.skip=handler;}};
  const status = {textContent:''};
  let fetchCount = 0;
  const context = {
    document:{body,getElementById:id=>id==='skip-animation'?skip:status,addEventListener:()=>{},querySelectorAll:()=>[],createElement:()=>({getContext:()=>({drawImage:()=>{}})})},
    window:{matchMedia:()=>({matches:reduced,addEventListener:()=>{}}),addEventListener:()=>{}},
    Image:class {naturalWidth=744;naturalHeight=1215;set src(value){queueMicrotask(()=>this.onload?.());}},
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
  assert.equal(h.skip.hidden,false);
  assert.equal(h.timers[0].delay,4000);h.timers[0].handler();
  assert.equal(h.classes.has('is-settling'),true);assert.equal(h.timers[1].delay,900);
  h.timers[1].handler();assert.equal(h.classes.has('is-complete'),true);
  assert.equal(h.skip.hidden,true);assert.equal(h.skip.textContent,'Skip animation');
});
test('skip before data arrives cannot start a delayed animation afterwards', async () => {
  let resolve;
  const pending=new Promise(done=>{resolve=done;});
  const h=harness({fetch:()=>pending});h.handlers.skip();
  resolve({ok:true,json:async()=>chartData()});await flush();
  assert.equal(h.classes.has('is-complete'),true);assert.equal(h.classes.has('is-illustrating'),false);assert.equal(h.timers.length,0);
});
test('failed illustration data leaves immediately readable content', async () => {
  const h=harness({fetch:async()=>{throw new Error('offline');}});await flush();
  assert.equal(h.classes.has('is-complete'),true);assert.equal(h.classes.has('is-illustrating'),false);
});
test('an unavailable chart response leaves readable copy and no looping intro', async () => {
  const h=harness({fetch:async()=>({ok:false})});await flush();
  assert.equal(h.classes.has('is-complete'),true);assert.equal(h.skip.hidden,true);
});
