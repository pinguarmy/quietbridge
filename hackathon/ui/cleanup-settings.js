(() => {
  const $ = id => document.getElementById(id);
  const form = $('cleanup-settings-form');
  if (!form) return;

  let token = '';
  let busy = false;
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
      $('cleanup-model').value = config.model || '';
      $('cleanup-enabled').checked = Boolean(config.enabled);
      status(config.configured ? 'A provider key is configured in backend memory.' : 'No provider key is configured.');
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
        body: JSON.stringify({base_url:$('cleanup-base-url').value.trim(), model:$('cleanup-model').value.trim(), api_key:key, enabled:$('cleanup-enabled').checked})
      });
      if (!response.ok) throw new Error();
      const result = await response.json();
      status(result.configured ? 'Settings saved. Provider key remains in backend memory.' : 'Settings saved. No provider key is configured.');
    } catch (_) {
      error('Could not save settings. Check the values and try again.'); status('');
    } finally {
      $('cleanup-api-key').value = '';
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
      status(result.latency_ms == null ? 'Connection successful.' : `Connection successful · ${result.latency_ms} ms`);
    } catch (_) {
      status(''); error('Connection test failed. Check the provider settings and try again.');
    } finally { setBusy(false); }
  });
  load();
})();
