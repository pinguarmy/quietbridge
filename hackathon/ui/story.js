(() => {
  'use strict';
  const body = document.body;
  const skip = document.getElementById('skip-animation');
  const status = document.getElementById('journey-status');
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const fields = [];
  const animations = [];
  let finished = false, timer = null;

  function finish() {
    finished = true;
    clearTimeout(timer);
    animations.forEach(animation => animation.cancel());
    body.classList.remove('is-illustrating', 'is-settling');
    body.classList.add('is-complete');
    fields.forEach(({cloud}) => { cloud.hidden = true; });
    status.textContent = 'Your words. Your choice.';
    skip.textContent = 'Animation skipped';
    skip.disabled = true;
  }
  skip.addEventListener('click', finish);
  document.addEventListener('keydown', event => { if (event.key === 'Escape') finish(); });
  reduced.addEventListener('change', event => { if (event.matches) finish(); });
  if (reduced.matches) { finish(); return; }

  function handCanvas(glyph, width, height) {
    const canvas = document.createElement('canvas');
    canvas.width = width; canvas.height = height;
    const context = canvas.getContext('2d');
    const pixels = context.createImageData(width, height);
    for (let i = 0; i < glyph.pixels.length; i++) {
      const value = glyph.pixels[i], offset = i * 4;
      pixels.data[offset] = value; pixels.data[offset + 1] = value;
      pixels.data[offset + 2] = value; pixels.data[offset + 3] = 255;
    }
    context.putImageData(pixels, 0, 0);
    return canvas;
  }

  function prepareField(element, glyphs) {
    const content = element.textContent;
    element.classList.add('hand-copy');
    const words = document.createElement('span');
    words.className = 'copy-words'; words.textContent = content;
    const cloud = document.createElement('span');
    cloud.className = 'hand-cloud'; cloud.setAttribute('aria-hidden', 'true');
    element.replaceChildren(words, cloud);
    const rect = element.getBoundingClientRect();
    const fontSize = parseFloat(getComputedStyle(element).fontSize);
    const size = Math.max(12, Math.min(34, Math.round(fontSize * .53)));
    const gap = fontSize > 40 ? 5 : 2;
    const width = Math.max(rect.width, size);
    let x = 0, y = 0, index = 0;
    const tiles = [];
    for (let i = 0; i < content.length; i++) {
      const character = content[i], letter = character.toUpperCase();
      if (/\s/.test(character)) { x += size * .43; continue; }
      if (!/[A-Z]/.test(letter)) { x += size * .22; continue; }
      if (x + size > width) { x = 0; y += size + gap; }
      const tile = document.createElement('span');
      tile.className = 'hand-tile';
      tile.style.setProperty('--tile-size', size + 'px');
      tile.style.left = x + 'px'; tile.style.top = y + 'px';
      const source = glyphs[letter];
      if (source) {
        const canvas = document.createElement('canvas');
        canvas.width = source.width; canvas.height = source.height;
        canvas.getContext('2d').drawImage(source, 0, 0);
        tile.append(canvas);
      } else {
        tile.classList.add('unavailable'); tile.textContent = '·';
        tile.title = 'No static letter image is available for ' + letter;
      }
      const range = document.createRange();
      range.setStart(words.firstChild, i); range.setEnd(words.firstChild, i + 1);
      const target = range.getBoundingClientRect();
      tiles.push({tile, x, y, targetX:target.left - rect.left + (target.width - size) / 2,
        targetY:target.top - rect.top + (target.height - size) / 2, delay:Math.min(index * 9, 180)});
      cloud.append(tile); x += size + gap; index++;
    }
    element.style.minHeight = Math.max(words.getBoundingClientRect().height, y + size) + 'px';
    fields.push({cloud, tiles});
  }

  function settle() {
    if (finished) return;
    body.classList.remove('is-illustrating'); body.classList.add('is-settling');
    fields.forEach(({tiles}) => tiles.forEach(({tile,x,y,targetX,targetY,delay}) => {
      animations.push(tile.animate([
        {transform:'translate(0, 0) scale(1)',opacity:1},
        {transform:'translate(' + (targetX-x) + 'px,' + (targetY-y) + 'px) scale(.5)',opacity:0}
      ], {duration:700,delay,easing:'cubic-bezier(.2,.7,.2,1)',fill:'forwards'}));
    }));
    timer = setTimeout(() => {
      finished = true; body.classList.remove('is-settling'); body.classList.add('is-complete');
      status.textContent = 'Your words. Your choice.';
      skip.textContent = 'Animation complete'; skip.disabled = true;
    }, 900);
  }

  async function begin() {
    try {
      const response = await fetch('/hand-letters.json');
      if (!response.ok) return;
      const data = await response.json();
      if (finished || reduced.matches) return;
      if (data.width !== 28 || data.height !== 28 || !data.glyphs) return;
      const glyphs = {};
      Object.entries(data.glyphs).forEach(([letter,glyph]) => {
        if (glyph.pixels?.length === 784 && glyph.pixels.every(value => Number.isInteger(value) && value >= 0 && value <= 255))
          glyphs[letter] = handCanvas(glyph, data.width, data.height);
      });
      document.querySelectorAll('[data-hand-text]').forEach(element => prepareField(element, glyphs));
      document.querySelectorAll('[data-sample]').forEach(element => {
        const source = glyphs[element.dataset.sample];
        if (!source) return;
        const canvas = document.createElement('canvas'); canvas.width=28; canvas.height=28;
        canvas.getContext('2d').drawImage(source,0,0); element.append(canvas); element.classList.add('has-sample');
      });
      body.classList.add('is-illustrating');
      status.textContent = 'A brief fingerspelling illustration. Skip whenever you like.';
      timer = setTimeout(settle, 4000);
    } catch (_) { finish(); }
  }
  window.addEventListener('resize', () => { if (body.classList.contains('is-illustrating') || body.classList.contains('is-settling')) finish(); });
  begin();
})();
