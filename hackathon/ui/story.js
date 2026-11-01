(() => {
  'use strict';
  const body = document.body;
  const skip = document.getElementById('skip-animation');
  const status = document.getElementById('journey-status');
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const fields = [];
  const samples = [];
  const animations = [];
  let finished = false, timer = null;

  function finish(focusCTA = false) {
    finished = true;
    clearTimeout(timer);
    animations.forEach(animation => animation.cancel());
    body.classList.remove('is-illustrating', 'is-settling');
    body.classList.add('is-complete');
    fields.forEach(({cloud}) => { cloud.hidden = true; });
    status.textContent = 'Your words. Your choice.';
    skip.hidden = true;
    skip.disabled = true;
    if (focusCTA) document.querySelector?.('.hero-actions a')?.focus?.({preventScroll:true});
  }
  skip.addEventListener('click', () => finish(true));
  document.addEventListener('keydown', event => { if (event.key === 'Escape') finish(); });
  reduced.addEventListener('change', event => { if (event.matches) finish(); });
  if (reduced.matches) { finish(); return; }

  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image); image.onerror = reject; image.src = url;
    });
  }

  function handCanvas(image, crop) {
    const canvas = document.createElement('canvas');
    canvas.width = 120; canvas.height = 120;
    const context = canvas.getContext('2d');
    const [x,y,width,height] = crop, scale = Math.min(112 / width,112 / height);
    context.drawImage(image,x,y,width,height,(120-width*scale)/2,(120-height*scale)/2,width*scale,height*scale);
    return canvas;
  }

  function addFaces(card, source, letter) {
    const front = document.createElement('span'); front.className = 'flap-front';
    if (source) {
      const canvas = document.createElement('canvas'); canvas.width=120; canvas.height=120;
      canvas.getContext('2d').drawImage(source,0,0); front.append(canvas);
    } else { front.textContent = '·'; front.classList.add('unavailable'); }
    const back = document.createElement('span'); back.className = 'flap-back'; back.textContent = letter;
    card.append(front,back);
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
    const size = Math.max(24, Math.min(48, Math.round(fontSize * .65)));
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
      tile.className = 'hand-tile flip-card';
      tile.style.setProperty('--tile-size', size + 'px');
      tile.style.setProperty('--letter-size', Math.min(fontSize,size*.8) + 'px');
      tile.style.left = x + 'px'; tile.style.top = y + 'px';
      const source = glyphs[letter];
      addFaces(tile,source,character);
      if (!source) tile.title = 'No letter image is available for ' + letter;
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
        {transform:'translate(0,0) rotateX(0deg)',opacity:1,offset:0},
        {transform:'translate(' + ((targetX-x)*.65) + 'px,' + ((targetY-y)*.65) + 'px) rotateX(-90deg)',opacity:1,offset:.44},
        {transform:'translate(' + (targetX-x) + 'px,' + (targetY-y) + 'px) rotateX(-180deg)',opacity:1,offset:.8},
        {transform:'translate(' + (targetX-x) + 'px,' + (targetY-y) + 'px) rotateX(-180deg)',opacity:0,offset:1}
      ], {duration:720,delay,easing:'cubic-bezier(.25,.5,.3,1)',fill:'forwards'}));
    }));
    samples.forEach((card,index) => animations.push(card.animate([{transform:'rotateX(0deg)'},{transform:'rotateX(-180deg)'}],{duration:650,delay:index*40,easing:'cubic-bezier(.25,.5,.3,1)',fill:'forwards'})));
    timer = setTimeout(() => {
      finished = true; body.classList.remove('is-settling'); body.classList.add('is-complete');
      status.textContent = 'Your words. Your choice.';
      skip.hidden = true; skip.disabled = true;
    }, 900);
  }

  async function begin() {
    try {
      const response = await fetch('/hand-letters.json');
      if (!response.ok) { finish(); return; }
      const data = await response.json();
      if (finished || reduced.matches) return;
      if (data.kind !== 'bsl-chart' || !data.glyphs || !data.image) { finish(); return; }
      const chart = await loadImage(data.image);
      if (finished || reduced.matches) return;
      if (chart.naturalWidth !== data.width || chart.naturalHeight !== data.height) { finish(); return; }
      const glyphs = {};
      Object.entries(data.glyphs).forEach(([letter,glyph]) => {
        const crop = glyph.crop;
        if (Array.isArray(crop) && crop.length === 4 && crop.every(Number.isInteger) && crop[0] >= 0 && crop[1] >= 0 && crop[2] > 0 && crop[3] > 0 && crop[0]+crop[2] <= data.width && crop[1]+crop[3] <= data.height)
          glyphs[letter] = handCanvas(chart,crop);
      });
      if (!Object.keys(glyphs).length) { finish(); return; }
      document.querySelectorAll('[data-hand-text]').forEach(element => prepareField(element, glyphs));
      document.querySelectorAll('[data-sample]').forEach(element => {
        const source = glyphs[element.dataset.sample];
        if (!source) return;
        const card = document.createElement('div'); card.className='sample-flipper flip-card';
        addFaces(card,source,element.dataset.sample); element.replaceChildren(card);
        element.classList.add('has-sample'); samples.push(card);
      });
      body.classList.add('is-illustrating');
      skip.hidden = false;
      status.textContent = 'A brief fingerspelling illustration. Skip whenever you like.';
      timer = setTimeout(settle, 4000);
    } catch (_) { finish(); }
  }
  window.addEventListener('resize', () => { if (body.classList.contains('is-illustrating') || body.classList.contains('is-settling')) finish(); });
  begin();
})();
