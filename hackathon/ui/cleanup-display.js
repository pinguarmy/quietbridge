(() => {
  const message = document.getElementById('message');
  const reason = document.getElementById('reason');
  if (!message || !reason) return;

  const panel = document.createElement('div');
  panel.id = 'cleanup-display';
  panel.className = 'alternatives';
  panel.setAttribute('aria-live', 'polite');
  reason.insertAdjacentElement('afterend', panel);

  window.renderCleanup = state => {
    const cleanup = state?.cleanup;
    const current = cleanup && cleanup.source_revision === state.revision;
    const pending = current && (cleanup.status === 'pending' || cleanup.pending === true);
    message.classList.toggle('cleanup-pending', Boolean(pending));
    panel.replaceChildren();
    if (!current) return;

    if (pending) {
      const wait = document.createElement('p');
      wait.className = 'hint';
      wait.textContent = '◌ Polishing suggestion…';
      panel.append(wait);
    }
    if (typeof cleanup.suggestion === 'string' && cleanup.suggestion.trim()) {
      const label = document.createElement('p');
      label.className = 'hint';
      label.textContent = 'Hint-assisted welcome suggestion · not raw lip recognition';
      const raw = document.createElement('details');
      const summary = document.createElement('summary');
      summary.textContent = 'Original lip recognition (may be incorrect)';
      const original = document.createElement('p');
      original.textContent = state.session_text || (state.candidates || []).join(' / ');
      raw.append(summary, original);
      message.textContent = cleanup.suggestion;
      // Displaying an AI suggestion must never leave Speak bound to different raw words.
      if (state.decision?.text !== cleanup.suggestion && document.getElementById('speak')) {
        document.getElementById('speak').disabled = true;
        const speakLabel = document.getElementById('speak-label');
        if (speakLabel) speakLabel.textContent = 'Select the welcome suggestion first';
      }
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'alternative';
      button.textContent = cleanup.suggestion;
      button.setAttribute('aria-label', 'Use optional AI text suggestion');
      button.onclick = () => {
        if (state.revision === cleanup.source_revision && typeof window.requestAction === 'function') {
          window.requestAction('confirm', {revision:cleanup.source_revision, text:cleanup.suggestion});
        }
      };
      panel.append(label, button, raw);
    }
  };
})();
