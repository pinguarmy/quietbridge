const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

test('new output animates once, polling stays still, and decoding alone shows progress', () => {
  const nodes = {}, animations = [];
  let reduced = false;
  for (const id of ['message','message-dock','message-progress','decision-status','reason','speak','speak-label']) {
    nodes[id] = {textContent:'', hidden:false, classList:{toggle(){}}, setAttribute(name,value){this[name]=value;},
      animate(){const animation={cancelled:false,cancel(){this.cancelled=true;}};animations.push(animation);return animation;}};
  }
  const context = vm.createContext({document:{getElementById:id=>nodes[id] || null,querySelectorAll:()=>[],addEventListener(){}},
    window:{addEventListener(){},matchMedia:()=>({matches:reduced})},fetch:()=>new Promise(()=>{}),setTimeout(){}});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../ui/app.js'),'utf8'),context);
  vm.runInContext("state={decision:{status:'accepted'},phase:'idle',speech_available:true}; chosenText='Hello'; updateMessage(); updateMessage();",context);
  assert.equal(nodes.message.textContent,'Hello');assert.equal(animations.length,1);
  vm.runInContext("chosenText='One moment'; updateMessage();",context);
  assert.equal(animations.length,2);assert.equal(animations[0].cancelled,true);
  reduced=true;
  vm.runInContext("chosenText='Yes'; updateMessage();",context);
  assert.equal(animations.length,2);assert.equal(nodes.message.textContent,'Yes');
  vm.runInContext("chosenText=null; state.phase='decoding'; updateMessage();",context);
  assert.equal(nodes['message-progress'].hidden,false);assert.equal(nodes['message-dock']['aria-busy'],'true');
  vm.runInContext("state.phase='idle'; updateMessage();",context);
  assert.equal(nodes['message-progress'].hidden,true);assert.equal(nodes.message.textContent,'');
});
