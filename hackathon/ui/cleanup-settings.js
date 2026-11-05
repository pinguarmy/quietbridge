(() => {
  const $ = id => document.getElementById(id);
  const form = $('cleanup-settings-form');
  if (!form) return;

  let token = '';
  let busy = false, dirty = false;
  form.addEventListener('input', () => { dirty = true; });
  window.addEventListener('beforeunload', event => { if (dirty) { event.preventDefault(); event.returnValue = ''; } });
  const status = message => { $('cleanup-settings-status').textContent = message; };
  const error = message => { $('cleanup-settings-error').textContent = message; };
  const setBusy = value => {
    busy = value;
    $('cleanup-save').disabled = value;
    $('cleanup-test').disabled = value;
  };
  async function load() {
    try {
      const stateResponse = await fetch('/api/state');
      if (!stateResponse.ok) throw new Error();
      const state = await stateResponse.json();
      token = state.token || '';
      const response = await fetch('/api/cleanup/config');
      if (!response.ok) throw new Error();
      const config = await response.json();
      $('cleanup-base-url').value = config.base_url || '';
      $('cleanup-model').value = config.model || 'gpt-6-luna';
      $('cleanup-tier').value = config.service_tier || 'default';
      $('cleanup-enabled').checked = Boolean(config.enabled);
      $('cleanup-api-key').placeholder = config.configured ? 'Key held in memory; leave blank to keep it' : 'Paste your provider API key';
      status(config.configured ? 'API key is saved. No need to enter it again after refreshing. Restarting the backend clears it.' : 'No API key saved yet.');
    } catch (_) {
      error('Could not load AI cleanup settings. Please refresh and try again.');
    }
  }
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (busy) return;
    setBusy(true); error(''); status('Saving…');
    const key = $('cleanup-api-key').value;
    try {
      const response = await fetch('/api/cleanup/config', {
        method: 'POST',
        headers: {'Content-Type':'application/json', 'X-Lipflow-Token':token},
        body: JSON.stringify({base_url:$('cleanup-base-url').value.trim(), model:$('cleanup-model').value.trim(), api_key:key, enabled:$('cleanup-enabled').checked, service_tier:$('cleanup-tier').value})
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Could not save preferences');
      $('cleanup-api-key').placeholder = result.configured ? 'Key saved — leave blank to keep it' : 'Paste your OpenAI API key';
      status(result.configured ? 'Saved. API key is configured; leave the password field blank unless replacing it.' : 'Settings saved. No API key saved yet.');
      $('cleanup-api-key').value = '';
      dirty = false;
    } catch (failure) {
      error(`${failure.message}. Your entries are still here. Correct them and try again.`); status('');
    } finally {
      setBusy(false);
    }
  });
  $('cleanup-test').addEventListener('click', async () => {
    if (busy) return;
    setBusy(true); error(''); status('Testing connection…');
    try {
      const response = await fetch('/api/cleanup/test', {
        method: 'POST', headers: {'Content-Type':'application/json', 'X-Lipflow-Token':token}, body: '{}'
      });
      const result = await response.json();
      if (!response.ok) throw new Error();
      const statusText = `Connection ${String(result.status || 'successful')}`;
      const latency = result.latency_ms ?? result.latency;
      status(latency == null ? `${statusText}.` : `${statusText} · ${latency} ms`);
    } catch (_) {
      status(''); error('Connection test failed. Check the provider settings and try again.');
    } finally { setBusy(false); }
  });
  load();
})();
