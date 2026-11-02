const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

test('gesture intent template loads, saves, clears and validates', async () => {
  const html = fs.readFileSync(path.join(__dirname, '..', 'lab.html'), 'utf8');
  const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  new vm.Script(script);
  const render = script.slice(script.indexOf('function renderSetupView()'), script.indexOf("$('btnAddGesture').onclick"));
  const save = script.slice(script.indexOf("$('btnSaveProfile').onclick=async()=>"), script.indexOf("$('btnExitSetup').onclick"));
  let sent, templateInput;
  const alerts = [], rows = [];
  const fields = {
    emoji: {value: '✋'}, label: {value: 'HELLO'}, display_name: {value: 'Hello'},
    concept: {value: 'hello'}, description: {value: 'Wave'}, intent_template: {value: ''}
  };
  const row = {querySelector(selector) { return fields[selector.match(/"(.*?)"/)[1]]; }};
  const tbody = {
    set innerHTML(value) {},
    appendChild(tr) { templateInput = tr.input; },
    querySelectorAll(selector) { return selector === 'tr' ? rows : []; }
  };
  const prompts = {
    innerHTML: '', appendChild() {},
    querySelectorAll(selector) { return selector === 'tr' ? [{querySelector() { return {value: 'Hello'}; }}] : []; }
  };
  const elements = {profileDisplayName: {value: ''}, gestureTbody: tbody, promptTbody: prompts, btnSaveProfile: {}};
  const context = {
    document: {createElement() { return {innerHTML: '', input: {value: ''}, querySelector() { return this.input; }}; }},
    $: id => elements[id],
    activeProfile: {id: 'p', name: 'P', gestures: [{label: 'HELLO', intent_template: 'Say "{gesture}" & go'}], lip_prompts: []},
    action: async (path, data) => { sent = data; return context.activeProfile; },
    loadProfiles: async () => {}, alert: message => alerts.push(message), setStage: () => {}
  };
  vm.createContext(context);
  vm.runInContext(render + save, context);
  context.renderSetupView();
  assert.equal(templateInput.value, 'Say "{gesture}" & go');
  delete context.activeProfile.gestures[0].intent_template;
  context.renderSetupView();
  assert.equal(templateInput.value, '');

  rows.push(row);
  fields.intent_template.value = 'I need {gesture}';
  await elements.btnSaveProfile.onclick();
  assert.equal(sent.gestures[0].intent_template, 'I need {gesture}');
  fields.intent_template.value = '';
  await elements.btnSaveProfile.onclick();
  assert.ok(!('intent_template' in sent.gestures[0]));
  for (const invalid of ['invalid', '{gesture} {gesture}']) {
    sent = null;
    fields.intent_template.value = invalid;
    await elements.btnSaveProfile.onclick();
    assert.equal(sent, null);
    assert.match(alerts.at(-1), /exactly one/);
  }
});
