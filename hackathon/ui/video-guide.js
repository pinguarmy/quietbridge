(() => {
  'use strict';
  const input = document.getElementById('demo-video-file');
  const video = document.getElementById('demo-video');
  const empty = document.getElementById('demo-video-empty');
  const status = document.getElementById('demo-video-status');
  const clear = document.getElementById('clear-demo-video');
  if (!input || !video) return;
  let objectUrl = null;
  function reset() {
    video.pause(); video.removeAttribute('src'); video.load(); video.hidden = true;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null; empty.hidden = false; clear.hidden = true; input.value = ''; status.textContent = '';
  }
  input.addEventListener('change', () => {
    const file = input.files?.[0];
    if (!file) return;
    reset();
    if (file.type && !file.type.startsWith('video/')) {
      status.textContent = 'Choose a video file to preview.'; return;
    }
    objectUrl = URL.createObjectURL(file);
    video.src = objectUrl; video.hidden = false; empty.hidden = true; clear.hidden = false;
    status.textContent = file.name; video.load();
  });
  video.addEventListener('error', () => {
    if (!objectUrl) return;
    reset(); status.textContent = 'This browser cannot play that video. Try another file.';
  });
  clear.addEventListener('click', reset);
  window.addEventListener('pagehide', () => { if (objectUrl) URL.revokeObjectURL(objectUrl); });
})();
