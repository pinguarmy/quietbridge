(() => {
  'use strict';
  document.documentElement.classList.add('story-pending');
  function start() {
  const body = document.body;
  const hands = document.getElementById('wordmark-hands');
  const next = document.getElementById('story-next');
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const animations = [];
  let finished = false, timer = null, watchdog = null;
  function finish() {
    finished = true;
    clearTimeout(timer); clearTimeout(watchdog);
    animations.forEach(animation => animation.cancel());
    body.classList.remove('is-loading', 'is-illustrating', 'is-settling');
    body.classList.add('is-complete');
    hands.hidden = true; next.hidden = false;
    document.documentElement.classList.remove('story-pending');
  }
  document.addEventListener('keydown', event => { if (event.key === 'Escape') finish(); });
  reduced.addEventListener('change', event => { if (event.matches) finish(); });
  if (reduced.matches) { finish(); return; }
  body.classList.add('is-loading'); next.hidden = true;
  // Unavailable or stalled assets must never strand the visitor.
  watchdog = setTimeout(finish, 10000);
  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image); image.onerror = reject; image.src = url;
    });
  }
  function card(image, crop, letter) {
    const tile = document.createElement('span'); tile.className = 'wordmark-tile';
    const flipper = document.createElement('span'); flipper.className = 'wordmark-flipper flip-card';
    const front = document.createElement('span'); front.className = 'flap-front';
    const canvas = document.createElement('canvas'); canvas.width = 160; canvas.height = 160;
    const [x,y,width,height] = crop, scale = Math.min(150/width,150/height);
    canvas.getContext('2d').drawImage(image,x,y,width,height,(160-width*scale)/2,(160-height*scale)/2,width*scale,height*scale);
    front.append(canvas);
    const back = document.createElement('span'); back.className = 'flap-back'; back.textContent = letter;
    flipper.append(front,back); tile.append(flipper); hands.append(tile);
    return flipper;
  }
  async function begin() {
    try {
      const response = await fetch('/hand-letters.json');
      if (!response.ok) { finish(); return; }
      const data = await response.json();
      if (finished) return;
      if (data.kind !== 'bsl-chart' || !data.glyphs || !data.image) { finish(); return; }
      const image = await loadImage(data.image);
      if (finished) return;
      if (image.naturalWidth !== data.width || image.naturalHeight !== data.height) { finish(); return; }
      const flippers = [];
      for (const letter of 'QuietBridge') {
        const crop = data.glyphs[letter.toUpperCase()]?.crop;
        if (!Array.isArray(crop) || crop.length !== 4 || !crop.every(Number.isInteger) || crop[0]<0 || crop[1]<0 || crop[2]<=0 || crop[3]<=0 || crop[0]+crop[2]>data.width || crop[1]+crop[3]>data.height) { finish(); return; }
        flippers.push(card(image,crop,letter));
      }
      body.classList.remove('is-loading'); body.classList.add('is-illustrating');
      timer = setTimeout(() => {
        if (finished) return;
        body.classList.remove('is-illustrating'); body.classList.add('is-settling');
        try {
          const word = document.querySelector?.('.wordmark-text');
          flippers.forEach((tile,index) => {
            let destination = 'rotateX(-180deg)';
            if (word?.firstChild && document.createRange) {
              const range = document.createRange();
              range.setStart(word.firstChild,index); range.setEnd(word.firstChild,index+1);
              const target = range.getBoundingClientRect(), origin = tile.getBoundingClientRect();
              const dx = target.left + target.width/2 - origin.left - origin.width/2;
              const dy = target.top + target.height/2 - origin.top - origin.height/2;
              const back = tile.querySelector('.flap-back');
              const scale = parseFloat(getComputedStyle(word).fontSize)/parseFloat(getComputedStyle(back).fontSize);
              destination = `translate(${dx}px,${dy}px) rotateX(-180deg) scale(${scale})`;
            }
            animations.push(tile.animate([{transform:'rotateX(0deg)',opacity:1},{transform:destination,opacity:1,offset:.82},{transform:destination,opacity:0}],{duration:800,delay:index*12,easing:'cubic-bezier(.25,.5,.3,1)',fill:'forwards'}));
          });
          timer = setTimeout(finish, 980);
        } catch (_) { finish(); }
      },4000);
    } catch (_) { finish(); }
  }
  begin();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, {once:true});
  else start();
})();
