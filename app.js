// ── Constants ────────────────────────────────────────────────────────────────
const STEPS  = 320;
const SURF_N = 72;
const TUBE_SEGMENTS_PER_STEP = 4;
const TUBE_RADIAL_SEGMENTS = 10;
const COLORS_HEX = { sgd:'#ff4fd8', mom:'#00d4ff', adam:'#7cff4f' };
const COLORS_NUM  = { sgd:0xff4fd8, mom:0x00d4ff, adam:0x7cff4f };
const TRACE_RADIUS = { sgd:0.033, mom:0.041, adam:0.036 };
const TRACE_OPACITY = { sgd:1.0, mom:1.0, adam:1.0 };
const NAMES       = { sgd:'SGD', mom:'SGD + Momentum', adam:'Adam' };

// ── Loss function ─────────────────────────────────────────────────────────────
// A long, rocky ravine: the valley floor winds toward the global minimum at (0,0).
// The squared ripple terms make the surface visually rugged without moving the true minimum.
function valleyY(x) { return 0.55 * Math.sin(1.25 * x); }
function loss(x, y) {
  const r = y - valleyY(x);
  const radial = x*x + y*y;
  const rocks1 = Math.sin(3.2*x) * Math.sin(2.6*y);
  const rocks2 = Math.sin(6.0*x + 1.7*y) * Math.cos(4.2*y);
  return 0.13*x*x + 3.2*r*r + 0.030*radial*rocks1*rocks1 + 0.012*radial*rocks2*rocks2;
}
// Numerical gradients keep the demo easy to modify: edit loss(), and the optimizers still follow it.
function gradX(x,y) { const h = 1e-4; return (loss(x+h,y)-loss(x-h,y))/(2*h); }
function gradY(x,y) { const h = 1e-4; return (loss(x,y+h)-loss(x,y-h))/(2*h); }
const LOSS_MAX = 42;
function clampL(l) { return Math.min(l, LOSS_MAX); }

// World→Three: (wx, wy, loss) → Vector3. Loss goes up (y axis).
function w2v(wx, wy, l) {
  return new THREE.Vector3(wx, clampL(l) * 3 / LOSS_MAX + 0.035, wy);
}

// ── Presets ───────────────────────────────────────────────────────────────────
const PRESETS = { A:{x:3.65,y:1.75}, B:{x:-3.45,y:-1.55}, C:{x:3.2,y:0.05} };
let startPos = {...PRESETS.A};
let paths = {}, step = 0, animProgress = 0, isRunning = false, animFrame = null, lastTickTime = 0, lastUiStep = -1;

// ── Three.js ──────────────────────────────────────────────────────────────────
const container = document.getElementById('three-container');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x0d0f14, 1);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0d0f14, 0.035);
const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 200);

// Lighting
scene.add(new THREE.AmbientLight(0xffffff, 0.38));
const dir1 = new THREE.DirectionalLight(0xffffff, 1.35);
dir1.position.set(4, 8, 4); scene.add(dir1);
const dir2 = new THREE.DirectionalLight(0x5599ff, 0.38);
dir2.position.set(-5, 2, -4); scene.add(dir2);

// ── Manual orbit controls ─────────────────────────────────────────────────────
let orbitTheta  = Math.PI / 4;
let orbitPhi    = 0.55;
let orbitRadius = 14;
let orbitTarget = new THREE.Vector3(0, 1.0, 0);
let dragging = false, dragBtn = -1;
let lastMouse = {x:0,y:0};
let lastTouchDist = 0;

function applyOrbit() {
  const sp = Math.sin(orbitPhi), cp = Math.cos(orbitPhi);
  const st = Math.sin(orbitTheta), ct = Math.cos(orbitTheta);
  camera.position.set(
    orbitTarget.x + orbitRadius * cp * ct,
    orbitTarget.y + orbitRadius * sp,
    orbitTarget.z + orbitRadius * cp * st
  );
  camera.lookAt(orbitTarget);
}

const cvs = renderer.domElement;

cvs.addEventListener('mousedown', e => {
  dragging = true; dragBtn = e.button;
  lastMouse = {x:e.clientX, y:e.clientY};
  document.getElementById('dragHint').style.opacity = '0';
});
window.addEventListener('mousemove', e => {
  if (!dragging) return;
  const dx = e.clientX - lastMouse.x, dy = e.clientY - lastMouse.y;
  lastMouse = {x:e.clientX, y:e.clientY};
  if (dragBtn === 0) {
    orbitTheta += dx * 0.007;
    orbitPhi = Math.max(-1.45, Math.min(1.45, orbitPhi + dy * 0.007));
  } else if (dragBtn === 2) {
    const right = new THREE.Vector3().crossVectors(
      camera.getWorldDirection(new THREE.Vector3()), camera.up).normalize();
    orbitTarget.addScaledVector(right, -dx * 0.006);
    orbitTarget.addScaledVector(camera.up, dy * 0.006);
  }
  applyOrbit();
});
window.addEventListener('mouseup', () => { dragging = false; });
cvs.addEventListener('wheel', e => {
  e.preventDefault();
  orbitRadius = Math.max(2.5, Math.min(22, orbitRadius + e.deltaY * 0.012));
  applyOrbit();
}, { passive: false });
cvs.addEventListener('contextmenu', e => e.preventDefault());

// Touch
cvs.addEventListener('touchstart', e => {
  if (e.touches.length === 1) {
    dragging = true; dragBtn = 0;
    lastMouse = {x:e.touches[0].clientX, y:e.touches[0].clientY};
    document.getElementById('dragHint').style.opacity = '0';
  } else if (e.touches.length === 2) {
    dragging = false;
    const dx = e.touches[0].clientX-e.touches[1].clientX;
    const dy = e.touches[0].clientY-e.touches[1].clientY;
    lastTouchDist = Math.sqrt(dx*dx+dy*dy);
  }
  e.preventDefault();
}, {passive:false});
cvs.addEventListener('touchmove', e => {
  if (e.touches.length === 1 && dragging) {
    const dx = e.touches[0].clientX-lastMouse.x, dy = e.touches[0].clientY-lastMouse.y;
    lastMouse = {x:e.touches[0].clientX, y:e.touches[0].clientY};
    orbitTheta += dx*0.007;
    orbitPhi = Math.max(-1.45, Math.min(1.45, orbitPhi + dy*0.007));
    applyOrbit();
  } else if (e.touches.length === 2) {
    const dx = e.touches[0].clientX-e.touches[1].clientX;
    const dy = e.touches[0].clientY-e.touches[1].clientY;
    const d = Math.sqrt(dx*dx+dy*dy);
    orbitRadius = Math.max(2.5, Math.min(22, orbitRadius*(lastTouchDist/d)));
    lastTouchDist = d; applyOrbit();
  }
  e.preventDefault();
}, {passive:false});
cvs.addEventListener('touchend', () => { dragging = false; });


// Floor grid gives the path shadows a stable x/y reference.
const grid = new THREE.GridHelper(8, 16, 0x33415f, 0x1f2937);
grid.position.y = 0;
grid.material.transparent = true;
grid.material.opacity = 0.22;
scene.add(grid);

// ── Surface mesh ──────────────────────────────────────────────────────────────
let surfMesh = null, wireMesh = null;

function buildSurface() {
  if (surfMesh) { scene.remove(surfMesh); surfMesh.geometry.dispose(); surfMesh.material.dispose(); }
  if (wireMesh) { scene.remove(wireMesh); wireMesh.geometry.dispose(); }

  const N = SURF_N;
  const geo = new THREE.BufferGeometry();
  const pos = [], col = [], idx = [];

  for (let i = 0; i <= N; i++) {
    for (let j = 0; j <= N; j++) {
      const wx = -4.0 + 8*i/N, wy = -2.5 + 5*j/N;
      const l = clampL(loss(wx, wy));
      const v = w2v(wx, wy, l);
      pos.push(v.x, v.y, v.z);
      // Color: deep blue at low loss, amber at high
      const t = l / LOSS_MAX;
      const r = 0.05 + t*0.55, g = 0.15 + (1-t)*0.35, b = 0.42 - t*0.22;
      col.push(r, g, b);
    }
  }

  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const a=i*(N+1)+j, b=a+1, c=a+N+1, d=c+1;
      idx.push(a,b,c, b,d,c);
    }
  }

  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('color',    new THREE.Float32BufferAttribute(col, 3));
  geo.setIndex(idx);
  geo.computeVertexNormals();

  const mat = new THREE.MeshPhongMaterial({ vertexColors:true, shininess:25, transparent:true, opacity:0.88, side:THREE.DoubleSide });
  surfMesh = new THREE.Mesh(geo, mat);
  scene.add(surfMesh);

  const wgeo = new THREE.WireframeGeometry(geo);
  const wmat = new THREE.LineBasicMaterial({ color:0xffffff, transparent:true, opacity:0.045 });
  wireMesh = new THREE.LineSegments(wgeo, wmat);
  scene.add(wireMesh);
}

// ── Path objects ──────────────────────────────────────────────────────────────
let pathObjs = { sgd:[], mom:[], adam:[] };
let markerObjs = [];

function clearScene3D() {
  ['sgd','mom','adam'].forEach(k => {
    pathObjs[k].forEach(o => { scene.remove(o); if(o.geometry) o.geometry.dispose(); if(o.material) { if(o.material.map) o.material.map.dispose(); o.material.dispose(); } });
    pathObjs[k] = [];
  });
  markerObjs.forEach(o => {
    scene.remove(o);
    o.traverse?.(child => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) { if (child.material.map) child.material.map.dispose(); child.material.dispose(); }
    });
    if(o.geometry) o.geometry.dispose();
    if(o.material) { if(o.material.map) o.material.map.dispose(); o.material.dispose(); }
  });
  markerObjs = [];
}

function makeDot(pos, color, size=8) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute([pos.x, pos.y, pos.z], 3));
  const mat = new THREE.PointsMaterial({ color, size, sizeAttenuation: false });
  const pts = new THREE.Points(geo, mat);
  scene.add(pts);
  return pts;
}



function makeSphere(pos, color, radius=0.12) {
  const geo = new THREE.SphereGeometry(radius, 24, 16);
  const mat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.35, roughness: 0.35 });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.copy(pos);
  scene.add(mesh);
  return mesh;
}

function makeLabelSprite(text, pos) {
  const canvas = document.createElement('canvas');
  canvas.width = 512; canvas.height = 160;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle = 'rgba(13,15,20,0.86)';
  ctx.strokeStyle = 'rgba(255,159,64,0.9)';
  ctx.lineWidth = 5;
  roundRect(ctx, 18, 24, 476, 92, 18);
  ctx.fill(); ctx.stroke();
  ctx.font = '700 34px JetBrains Mono, monospace';
  ctx.fillStyle = '#ffb45c';
  ctx.textAlign = 'center';
  ctx.fillText(text, 256, 79);
  ctx.font = '500 18px JetBrains Mono, monospace';
  ctx.fillStyle = 'rgba(255,255,255,0.72)';
  ctx.fillText('global minimum θ*', 256, 106);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.position.copy(pos);
  sprite.scale.set(2.0, 0.62, 1);
  scene.add(sprite);
  return sprite;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x+r, y);
  ctx.arcTo(x+w, y, x+w, y+h, r);
  ctx.arcTo(x+w, y+h, x, y+h, r);
  ctx.arcTo(x, y+h, x, y, r);
  ctx.arcTo(x, y, x+w, y, r);
  ctx.closePath();
}

function makeVerticalStem(pos, color) {
  const pts = [new THREE.Vector3(pos.x, 0.02, pos.z), new THREE.Vector3(pos.x, pos.y + 0.58, pos.z)];
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineDashedMaterial({ color, dashSize: 0.09, gapSize: 0.06, transparent:true, opacity:0.85 });
  const line = new THREE.Line(geo, mat);
  line.computeLineDistances();
  scene.add(line);
  return line;
}

function makeBeacon(pos, color) {
  const group = new THREE.Group();

  const stemGeo = new THREE.CylinderGeometry(0.008, 0.008, 0.55, 12);
  const stemMat = new THREE.MeshBasicMaterial({ color, transparent:true, opacity:0.45 });
  const stem = new THREE.Mesh(stemGeo, stemMat);
  stem.position.set(pos.x, pos.y + 0.28, pos.z);
  group.add(stem);

  const ringGeo = new THREE.TorusGeometry(0.16, 0.008, 8, 36);
  const ringMat = new THREE.MeshBasicMaterial({ color, transparent:true, opacity:0.72 });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.position.set(pos.x, pos.y + 0.04, pos.z);
  ring.rotation.x = Math.PI / 2;
  group.add(ring);

  scene.add(group);
  return group;
}

function makeTube(points, color, radius=0.035, opacity=1.0) {
  if (points.length < 2) return null;
  const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.25);
  const geo = new THREE.TubeGeometry(curve, Math.max(8, points.length * 2), radius, 10, false);
  const mat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.9, roughness: 0.25, metalness: 0.05, transparent:true, opacity });
  const tube = new THREE.Mesh(geo, mat);
  scene.add(tube);
  return tube;
}

function makeShadowLine(points, color) {
  if (points.length < 2) return null;
  const floorPts = points.map(p => new THREE.Vector3(p.x, 0.012, p.z));
  const geo = new THREE.BufferGeometry().setFromPoints(floorPts);
  const mat = new THREE.LineBasicMaterial({ color, transparent:true, opacity:0.32 });
  const line = new THREE.Line(geo, mat);
  scene.add(line);
  return line;
}

function redraw3D(activePaths, upTo) {
  clearScene3D();
  const upto = (upTo === undefined) ? STEPS : upTo;

  // Small global minimum marker: identified in the header legend, not with a large floating label.
  const mp = w2v(0, 0, loss(0,0));
  markerObjs.push(makeSphere(mp, 0xff9f40, 0.075));
  markerObjs.push(makeBeacon(mp, 0xff9f40));

  // Start marker
  const sp2 = w2v(startPos.x, startPos.y, loss(startPos.x, startPos.y));
  markerObjs.push(makeSphere(sp2, 0xf0c060, 0.14));

  const togMap = {sgd:'togSGD', mom:'togMOM', adam:'togADAM'};
  ['sgd','mom','adam'].forEach(key => {
    if (!document.getElementById(togMap[key]).classList.contains('checked')) return;
    const path = activePaths[key];
    if (!path || path.length < 2) return;
    const n = Math.min(upto, path.length-1);
    const threePts = [];
    for (let i=0; i<=n; i++) threePts.push(w2v(path[i].x, path[i].y, path[i].l));

    // Bright 3D tube: WebGL ignores thick LineBasicMaterial linewidth in many browsers,
    // so tubes make the trajectory visible from every camera angle.
    const shadow = makeShadowLine(threePts, COLORS_NUM[key]);
    const tube = makeTube(threePts, COLORS_NUM[key], TRACE_RADIUS[key], TRACE_OPACITY[key]);
    const head = makeSphere(threePts[threePts.length - 1], COLORS_NUM[key], 0.115);
    if (shadow) pathObjs[key].push(shadow);
    if (tube) pathObjs[key].push(tube);
    pathObjs[key].push(head);
  });
}


// ── 2D Chart ─────────────────────────────────────────────────────────────────
const c2d = document.getElementById('c2d');
const ctx2 = c2d.getContext('2d');

function draw2D(activePaths, upTo) {
  const W = c2d.width, H = c2d.height;
  ctx2.clearRect(0, 0, W, H);
  const PAD = {l:34, r:8, t:6, b:20};
  const pW = W-PAD.l-PAD.r, pH = H-PAD.t-PAD.b;
  const upto = (upTo === undefined) ? STEPS : upTo;
  let maxL = 0.1;
  const togMap = {sgd:'togSGD', mom:'togMOM', adam:'togADAM'};
  ['sgd','mom','adam'].forEach(k => {
    if (!document.getElementById(togMap[k]).classList.contains('checked')) return;
    if (activePaths[k]) activePaths[k].slice(0,upto+1).forEach(p=>{if(p.l>maxL)maxL=p.l;});
  });

  [0,.25,.5,.75,1].forEach(t => {
    const yy = PAD.t + pH*(1-t);
    ctx2.strokeStyle='rgba(255,255,255,0.05)'; ctx2.lineWidth=0.5;
    ctx2.beginPath(); ctx2.moveTo(PAD.l,yy); ctx2.lineTo(W-PAD.r,yy); ctx2.stroke();
    ctx2.fillStyle='rgba(255,255,255,0.22)'; ctx2.font='9px JetBrains Mono,monospace'; ctx2.textAlign='right';
    ctx2.fillText((t*maxL).toFixed(1), PAD.l-4, yy+3);
  });

  ctx2.textAlign='center'; ctx2.fillStyle='rgba(255,255,255,0.18)'; ctx2.font='9px JetBrains Mono,monospace';
  ctx2.fillText('steps', PAD.l+pW/2, H-2);
  [0,80,160,240,320].forEach(s => { const px=PAD.l+pW*(s/STEPS); ctx2.fillText(s,px,H-PAD.b+12); });

  ['sgd','mom','adam'].forEach(key => {
    if (!document.getElementById(togMap[key]).classList.contains('checked')) return;
    const path = activePaths[key];
    if (!path || path.length < 2) return;
    ctx2.beginPath();
    for (let i=0; i<=Math.min(upto,path.length-1); i++) {
      const px=PAD.l+pW*(i/STEPS), py=PAD.t+pH*(1-Math.min(path[i].l,maxL)/maxL);
      i===0 ? ctx2.moveTo(px,py) : ctx2.lineTo(px,py);
    }
    ctx2.strokeStyle=COLORS_HEX[key]; ctx2.lineWidth=1.5; ctx2.stroke();
  });
}

// ── Optimizer math ────────────────────────────────────────────────────────────
function computePath(optimizer) {
  const lr=getLR(), beta=getBeta();
  let x=startPos.x, y=startPos.y, vx=0,vy=0, mx=0,my=0, vx2=0,vy2=0;
  const pts = [{x,y,l:clampL(loss(x,y))}];
  for (let i=0;i<STEPS;i++) {
    const gx=gradX(x,y), gy=gradY(x,y);
    if (optimizer==='sgd') {
      x-=lr*gx; y-=lr*gy;
    } else if (optimizer==='mom') {
      vx=beta*vx+lr*gx; vy=beta*vy+lr*gy; x-=vx; y-=vy;
    } else {
      const ep=1e-8,b1=beta,b2=0.999;
      mx=b1*mx+(1-b1)*gx; my=b1*my+(1-b1)*gy;
      vx2=b2*vx2+(1-b2)*gx*gx; vy2=b2*vy2+(1-b2)*gy*gy;
      const mhx=mx/(1-Math.pow(b1,i+1)), mhy=my/(1-Math.pow(b1,i+1));
      const vhx=vx2/(1-Math.pow(b2,i+1)), vhy=vy2/(1-Math.pow(b2,i+1));
      x-=lr*mhx/(Math.sqrt(vhx)+ep); y-=lr*mhy/(Math.sqrt(vhy)+ep);
    }
    x=Math.max(-4.0,Math.min(4.0,x)); y=Math.max(-2.5,Math.min(2.5,y));
    pts.push({x,y,l:clampL(loss(x,y))});
  }
  return pts;
}

function getLR()   { return parseInt(document.getElementById('lrSlider').value)*0.01; }
function getBeta() { return parseInt(document.getElementById('momSlider').value)*0.01; }
// Returns {framesPerStep, stepsPerFrame}
function getSpeedConfig() {
  const v = parseInt(document.getElementById('speedSlider').value);
  const cfg = {1:[8,1], 2:[4,1], 3:[2,1], 4:[1,1], 5:[1,1], 6:[1,2]};
  const [fpS, spF] = cfg[v] || [1,1];
  return { framesPerStep: fpS, stepsPerFrame: spF };
}

function updateLossReadouts(s) {
  const km={sgd:'lossSGD',mom:'lossMOM',adam:'lossADAM'};
  const tm={sgd:'togSGD',mom:'togMOM',adam:'togADAM'};
  ['sgd','mom','adam'].forEach(k => {
    const el=document.getElementById(km[k]);
    if (!document.getElementById(tm[k]).classList.contains('checked')||!paths[k]) { el.textContent='—'; return; }
    el.textContent=paths[k][Math.min(s,STEPS)].l.toFixed(4);
  });
}

function getInsight(s, ap) {
  const tm={sgd:'togSGD',mom:'togMOM',adam:'togADAM'};
  const active=['sgd','mom','adam'].filter(k=>document.getElementById(tm[k]).classList.contains('checked')&&ap[k]);
  if (!isRunning&&s===0) return 'Choose a <strong>start position</strong>, press <strong>Run</strong>. <strong>Drag the 3D surface</strong> to view from any angle.';
  if (s<5) return 'Early steps — all optimizers follow the <strong>steepest descent</strong> direction. Watch how paths diverge.';
  if (s<20) {
    if (active.includes('mom')) return '<strong>Momentum</strong> accumulates velocity like a ball rolling downhill — building speed in consistent directions, dampening cross-valley oscillations.';
    return 'Early convergence phase. Watch how each optimizer responds differently to the gradient.';
  }
  if (s<50) {
    if (active.includes('adam')&&active.includes('sgd')) {
      const al=ap.adam[s].l, sl=ap.sgd[s].l;
      if (al<sl*0.6) return '<strong>Adam</strong> pulls ahead by adapting per-parameter learning rates — small updates in steep dimensions, large updates in flat ones.';
    }
    if (active.includes('mom')&&active.includes('sgd')) {
      const ml=ap.mom[s].l, sl=ap.sgd[s].l;
      if (ml<sl*0.7) return 'Momentum\'s <strong>velocity accumulation</strong> has built enough inertia to slide through flat regions that stall vanilla SGD.';
    }
    return 'Mid-run: compare the <strong>loss curves</strong> below. Steeper descent = faster convergence.';
  }
  if (s<90) return 'Notice any <strong>oscillation</strong> across the valley? That\'s the hallmark of β too high or learning rate too large. Try reducing β or α.';
  const finals=active.map(k=>({k,l:ap[k][s].l})).sort((a,b)=>a.l-b.l);
  if (finals.length>0) {
    const w=finals[0];
    return `<strong>${NAMES[w.k]}</strong> converged fastest (loss ${w.l.toFixed(4)}). Try a different start or adjust hyperparameters to shift rankings.`;
  }
  return 'Run complete. Adjust hyperparameters and run again to compare.';
}

// ── Sim loop ──────────────────────────────────────────────────────────────────
function tick() {
  const { framesPerStep, stepsPerFrame } = getSpeedConfig();
  let advanced = false;

  slowTick++;
  if (slowTick >= framesPerStep) {
    slowTick = 0;
    step = Math.min(step + stepsPerFrame, STEPS);
    advanced = true;
  }

  if (advanced) {
    redraw3D(paths, step);
    draw2D(paths, step);
    updateLossReadouts(step);
    document.getElementById('stepLabel').textContent = step;
    document.getElementById('stepCounter').textContent = `step ${step} / ${STEPS}`;
    document.getElementById('progFill').style.width = (step/STEPS*100).toFixed(1)+'%';
    document.getElementById('insightBox').innerHTML = getInsight(step, paths);
  }

  if (step >= STEPS) {
    isRunning = false;
    document.getElementById('runBtn').textContent = '↺  RUN AGAIN';
    document.getElementById('runBtn').classList.remove('running');
    document.getElementById('progFill').style.width = '100%';
    return;
  }
  if (isRunning) animFrame = requestAnimationFrame(tick);
}

// ── Render loop ───────────────────────────────────────────────────────────────
function renderLoop() {
  requestAnimationFrame(renderLoop);
  renderer.render(scene, camera);
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetAll() {
  if (animFrame) cancelAnimationFrame(animFrame);
  isRunning=false; step=0; paths={}; slowTick=0;
  document.getElementById('runBtn').textContent='▶  RUN';
  document.getElementById('runBtn').classList.remove('running');
  document.getElementById('stepLabel').textContent='0';
  document.getElementById('stepCounter').textContent=`step 0 / ${STEPS}`;
  document.getElementById('progFill').style.width='0%';
  updateLossReadouts(0);
  document.getElementById('insightBox').innerHTML = getInsight(0,{});
  ctx2.clearRect(0,0,c2d.width,c2d.height);
  redraw3D({},0);
}

document.getElementById('resetBtn').addEventListener('click', resetAll);

document.getElementById('runBtn').addEventListener('click', () => {
  if (isRunning) {
    cancelAnimationFrame(animFrame); isRunning=false;
    document.getElementById('runBtn').textContent='▶  RESUME';
    document.getElementById('runBtn').classList.remove('running');
    return;
  }
  if (step===0||step>=STEPS) {
    paths={};
    const tm={sgd:'togSGD',mom:'togMOM',adam:'togADAM'};
    ['sgd','mom','adam'].forEach(k=>{
      if(document.getElementById(tm[k]).classList.contains('checked')) paths[k]=computePath(k);
    });
    step=0; slowTick=0;
  }
  isRunning=true;
  document.getElementById('runBtn').textContent='⏸  PAUSE';
  document.getElementById('runBtn').classList.add('running');
  animFrame=requestAnimationFrame(tick);
});

['A','B','C'].forEach(k=>{
  document.getElementById('pre'+k).addEventListener('click',()=>{
    document.querySelectorAll('.preset-btn').forEach(b=>b.classList.remove('active'));
    document.getElementById('pre'+k).classList.add('active');
    startPos={...PRESETS[k]}; resetAll();
  });
});

document.getElementById('lrSlider').addEventListener('input',()=>{
  document.getElementById('lrVal').textContent=getLR().toFixed(2);
});
document.getElementById('momSlider').addEventListener('input',()=>{
  document.getElementById('momVal').textContent=getBeta().toFixed(2);
});
document.getElementById('speedSlider').addEventListener('input',()=>{
  const v=parseInt(document.getElementById('speedSlider').value);
  const labels={1:'0.1×',2:'0.25×',3:'0.5×',4:'1×',5:'1.5×',6:'2×'};
  document.getElementById('speedVal').textContent=labels[v]||v+'×';
});

['togSGD','togMOM','togADAM'].forEach(id=>{
  document.getElementById(id).addEventListener('click', e => {
    e.preventDefault(); // prevent label from re-triggering the hidden checkbox
    const tog = document.getElementById(id);
    tog.classList.toggle('checked');
    if (!isRunning&&step>0) { redraw3D(paths,step); draw2D(paths,step); }
  });
});

function resizeAll() {
  const w=container.clientWidth, h=container.clientHeight;
  renderer.setSize(w,h);
  camera.aspect=w/h; camera.updateProjectionMatrix();
  c2d.width=c2d.parentElement.clientWidth-28; c2d.height=80;
  draw2D(paths,step);
}

window.addEventListener('resize',()=>{ resizeAll(); redraw3D(paths,step); });

// ── Init ──────────────────────────────────────────────────────────────────────
applyOrbit();
buildSurface();
resizeAll();
redraw3D({},0);
renderLoop();

setTimeout(()=>{ document.getElementById('dragHint').style.opacity='0'; }, 5000);
