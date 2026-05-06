// ── Constants ────────────────────────────────────────────────────────────────
const STEPS  = 320;
const SURF_N = 72;
const TUBE_RADIAL_SEGMENTS = 10;
const COLORS_HEX = { sgd:'#ff4fd8', mom:'#00d4ff', adam:'#7cff4f' };
const COLORS_NUM  = { sgd:0xff4fd8, mom:0x00d4ff, adam:0x7cff4f };
const TRACE_RADIUS = { sgd:0.033, mom:0.041, adam:0.036 };
const TRACE_OPACITY = { sgd:1.0, mom:1.0, adam:1.0 };
const NAMES       = { sgd:'SGD', mom:'SGD + Momentum', adam:'Adam' };
const DOMAIN = { xMin:-4.0, xMax:4.0, yMin:-2.5, yMax:2.5 };
const MAX_OPTIMIZER_STEP = 0.22;

// ── Loss landscapes ──────────────────────────────────────────────────────────
function ravineY(x) { return 0.55 * Math.sin(1.25 * x); }

const LANDSCAPES = {
  ravine: {
    label: 'Winding ravine',
    min: {x:0, y:0},
    lossMax: 42,
    defaults: { lr:8, beta:90 },
    lrScale: { sgd:1.0, mom:0.72, adam:1.45 },
    starts: {
      A:{label:'Rocky ridge', x:3.65, y:1.75},
      B:{label:'Opposite ridge', x:-3.45, y:-1.55},
      C:{label:'Ravine mouth', x:3.2, y:0.05}
    },
    loss(x, y) {
      const r = y - ravineY(x);
      const radial = x*x + y*y;
      const rocks1 = Math.sin(3.2*x) * Math.sin(2.6*y);
      const rocks2 = Math.sin(6.0*x + 1.7*y) * Math.cos(4.2*y);
      return 0.13*x*x + 3.2*r*r + 0.030*radial*rocks1*rocks1 + 0.012*radial*rocks2*rocks2;
    }
  },
  bowl: {
    label: 'Tilted bowl',
    min: {x:0, y:0},
    lossMax: 28,
    defaults: { lr:10, beta:85 },
    lrScale: { sgd:1.0, mom:1.0, adam:1.2 },
    starts: {
      A:{label:'Wide side', x:3.5, y:1.9},
      B:{label:'Steep wall', x:-3.2, y:2.1},
      C:{label:'Near floor', x:1.6, y:-0.9}
    },
    loss(x, y) {
      const u = 0.78*x + 0.63*y;
      const v = -0.63*x + 0.78*y;
      return 0.28*u*u + 2.35*v*v;
    }
  },
  ripples: {
    label: 'Rippled basin',
    min: {x:0, y:0},
    lossMax: 34,
    defaults: { lr:6, beta:80 },
    lrScale: { sgd:0.9, mom:0.85, adam:1.05 },
    starts: {
      A:{label:'Outer ring', x:3.55, y:-1.7},
      B:{label:'Side basin', x:-3.2, y:1.85},
      C:{label:'Noisy saddle', x:0.4, y:2.15}
    },
    loss(x, y) {
      const radial = x*x + y*y;
      const ripple = Math.sin(3.8*x) * Math.sin(3.2*y);
      const ring = Math.sin(2.4 * Math.sqrt(radial + 0.01));
      return 0.18*radial + 0.055*radial*ripple*ripple + 0.040*radial*ring*ring;
    }
  }
};

let activeLandscapeKey = 'ravine';
let LOSS_MAX = LANDSCAPES[activeLandscapeKey].lossMax;

function activeLandscape() { return LANDSCAPES[activeLandscapeKey]; }
function loss(x, y) { return activeLandscape().loss(x, y); }
// Numerical gradients keep the demo easy to modify: edit a landscape loss(), and the optimizers still follow it.
function gradX(x,y) { const h = 1e-4; return (loss(x+h,y)-loss(x-h,y))/(2*h); }
function gradY(x,y) { const h = 1e-4; return (loss(x,y+h)-loss(x,y-h))/(2*h); }
function clampL(l) { return Math.min(l, LOSS_MAX); }

// World→Three: (wx, wy, loss) → Vector3. Loss goes up (y axis).
function w2v(wx, wy, l) {
  return new THREE.Vector3(wx, clampL(l) * 3 / LOSS_MAX + 0.035, wy);
}

// ── Presets ───────────────────────────────────────────────────────────────────
let startKey = 'A';
let startPos = {...activeLandscape().starts[startKey]};
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
      const wx = DOMAIN.xMin + (DOMAIN.xMax - DOMAIN.xMin)*i/N;
      const wy = DOMAIN.yMin + (DOMAIN.yMax - DOMAIN.yMin)*j/N;
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
let pathVisuals = {};
let markerObjs = [];
const segmentMatrix = new THREE.Matrix4();
const segmentPosition = new THREE.Vector3();
const segmentScale = new THREE.Vector3();
const shadowEnd = new THREE.Vector3();

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
  pathVisuals = {};
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

function makeTubeInstances(points, color, radius=0.035, opacity=1.0) {
  const count = Math.max(0, points.length - 1);
  const geo = new THREE.CylinderGeometry(radius, radius, 1, TUBE_RADIAL_SEGMENTS, 1, false);
  const mat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 1.15, roughness: 0.22, metalness: 0.05, transparent:true, opacity });
  const mesh = new THREE.InstancedMesh(geo, mat, count);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

  const segments = [];
  for (let i=0; i<count; i++) {
    const delta = new THREE.Vector3().subVectors(points[i+1], points[i]);
    const length = delta.length();
    const dir = length > 1e-6 ? delta.clone().normalize() : new THREE.Vector3(0, 1, 0);
    const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    segments.push({ start: points[i].clone(), dir, length, quat });
    segmentMatrix.makeScale(0, 0, 0);
    mesh.setMatrixAt(i, segmentMatrix);
  }
  mesh.instanceMatrix.needsUpdate = true;
  scene.add(mesh);
  return { mesh, segments };
}

function makeShadowPath(points, color) {
  const floorPts = points.map(p => new THREE.Vector3(p.x, 0.012, p.z));
  const positions = new Float32Array(floorPts.length * 3);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setDrawRange(0, 0);
  const mat = new THREE.LineBasicMaterial({ color, transparent:true, opacity:0.32 });
  const line = new THREE.Line(geo, mat);
  scene.add(line);
  return { line, floorPts };
}

function pointAtProgress(path, progress) {
  const a = Math.max(0, Math.min(path.length - 1, Math.floor(progress)));
  const b = Math.min(path.length - 1, a + 1);
  const t = Math.max(0, Math.min(1, progress - a));
  const va = w2v(path[a].x, path[a].y, path[a].l);
  const vb = w2v(path[b].x, path[b].y, path[b].l);
  return va.lerp(vb, t);
}

function updateTrajectoryVisuals(progress) {
  ['sgd','mom','adam'].forEach(key => {
    const visual = pathVisuals[key];
    if (!visual) return;

    const clamped = Math.max(0, Math.min(STEPS, progress));
    visual.tube.segments.forEach((segment, i) => {
      const localProgress = Math.max(0, Math.min(1, clamped - i));
      if (localProgress > 0 && segment.length > 1e-6) {
        const visibleLength = segment.length * localProgress;
        segmentPosition.copy(segment.start).addScaledVector(segment.dir, visibleLength * 0.5);
        segmentScale.set(1, visibleLength, 1);
        segmentMatrix.compose(segmentPosition, segment.quat, segmentScale);
      } else {
        segmentMatrix.makeScale(0, 0, 0);
      }
      visual.tube.mesh.setMatrixAt(i, segmentMatrix);
    });
    visual.tube.mesh.instanceMatrix.needsUpdate = true;

    const wholeStep = Math.max(0, Math.min(Math.floor(clamped), visual.shadow.floorPts.length - 1));
    const shadowProgress = clamped - wholeStep;
    const shadowPositions = visual.shadow.line.geometry.attributes.position;
    for (let i=0; i<=wholeStep; i++) {
      const p = visual.shadow.floorPts[i];
      shadowPositions.setXYZ(i, p.x, p.y, p.z);
    }
    let shadowCount = wholeStep + 1;
    if (clamped > wholeStep && wholeStep < visual.shadow.floorPts.length - 1) {
      shadowEnd.copy(visual.shadow.floorPts[wholeStep]).lerp(visual.shadow.floorPts[wholeStep+1], shadowProgress);
      shadowPositions.setXYZ(wholeStep + 1, shadowEnd.x, shadowEnd.y, shadowEnd.z);
      shadowCount++;
    }
    visual.shadow.line.geometry.setDrawRange(0, shadowCount);
    shadowPositions.needsUpdate = true;

    visual.head.position.copy(pointAtProgress(visual.path, clamped));
  });
}

function redraw3D(activePaths, upTo) {
  clearScene3D();
  const upto = (upTo === undefined) ? STEPS : upTo;

  // Small global minimum marker: identified in the header legend, not with a large floating label.
  const min = activeLandscape().min;
  const mp = w2v(min.x, min.y, loss(min.x, min.y));
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
    const threePts = [];
    for (let i=0; i<path.length; i++) threePts.push(w2v(path[i].x, path[i].y, path[i].l));

    const tube = makeTubeInstances(threePts, COLORS_NUM[key], TRACE_RADIUS[key], TRACE_OPACITY[key]);
    const shadow = makeShadowPath(threePts, COLORS_NUM[key]);
    const head = makeSphere(threePts[0], COLORS_NUM[key], 0.115);
    pathObjs[key].push(tube.mesh, shadow.line);
    pathObjs[key].push(head);
    pathVisuals[key] = { tube, shadow, head, path };
  });
  updateTrajectoryVisuals(upto);
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
    const wholeStep = Math.max(0, Math.min(Math.floor(upto), path.length-1));
    for (let i=0; i<=wholeStep; i++) {
      const px=PAD.l+pW*(i/STEPS), py=PAD.t+pH*(1-Math.min(path[i].l,maxL)/maxL);
      i===0 ? ctx2.moveTo(px,py) : ctx2.lineTo(px,py);
    }
    if (upto > wholeStep && wholeStep < path.length-1) {
      const t = upto - wholeStep;
      const lossNow = path[wholeStep].l + (path[wholeStep+1].l - path[wholeStep].l) * t;
      const px=PAD.l+pW*(upto/STEPS), py=PAD.t+pH*(1-Math.min(lossNow,maxL)/maxL);
      ctx2.lineTo(px,py);
    }
    ctx2.strokeStyle=COLORS_HEX[key]; ctx2.lineWidth=1.5; ctx2.stroke();
  });
}

// ── Optimizer math ────────────────────────────────────────────────────────────
function boundedMove(x, y, dx, dy) {
  const stepLen = Math.hypot(dx, dy);
  if (stepLen > MAX_OPTIMIZER_STEP) {
    const scale = MAX_OPTIMIZER_STEP / stepLen;
    dx *= scale;
    dy *= scale;
  }

  const nextX = x - dx;
  const nextY = y - dy;
  const clampedX = Math.max(DOMAIN.xMin, Math.min(DOMAIN.xMax, nextX));
  const clampedY = Math.max(DOMAIN.yMin, Math.min(DOMAIN.yMax, nextY));
  return {
    x: clampedX,
    y: clampedY,
    dx: x - clampedX,
    dy: y - clampedY,
    hitX: clampedX !== nextX,
    hitY: clampedY !== nextY
  };
}

function computePath(optimizer) {
  const lr=getOptimizerLR(optimizer), beta=getBeta();
  let x=startPos.x, y=startPos.y, vx=0,vy=0, mx=0,my=0, vx2=0,vy2=0;
  const pts = [{x,y,l:clampL(loss(x,y))}];
  for (let i=0;i<STEPS;i++) {
    const gx=gradX(x,y), gy=gradY(x,y);
    let dx=0, dy=0;
    if (optimizer==='sgd') {
      dx=lr*gx; dy=lr*gy;
    } else if (optimizer==='mom') {
      vx=beta*vx+lr*gx; vy=beta*vy+lr*gy;
      dx=vx; dy=vy;
    } else {
      const ep=1e-8,b1=beta,b2=0.999;
      mx=b1*mx+(1-b1)*gx; my=b1*my+(1-b1)*gy;
      vx2=b2*vx2+(1-b2)*gx*gx; vy2=b2*vy2+(1-b2)*gy*gy;
      const mhx=mx/(1-Math.pow(b1,i+1)), mhy=my/(1-Math.pow(b1,i+1));
      const vhx=vx2/(1-Math.pow(b2,i+1)), vhy=vy2/(1-Math.pow(b2,i+1));
      dx=lr*mhx/(Math.sqrt(vhx)+ep);
      dy=lr*mhy/(Math.sqrt(vhy)+ep);
    }
    const next = boundedMove(x, y, dx, dy);
    x = next.x; y = next.y;
    if (optimizer === 'mom') {
      vx = next.dx;
      vy = next.dy;
      if (next.hitX) vx = 0;
      if (next.hitY) vy = 0;
    }
    pts.push({x,y,l:clampL(loss(x,y))});
  }
  return pts;
}

function getLR()   { return parseInt(document.getElementById('lrSlider').value)*0.01; }
function getBeta() { return parseInt(document.getElementById('momSlider').value)*0.01; }
function getOptimizerLR(optimizer) {
  const scale = activeLandscape().lrScale?.[optimizer] || 1;
  return getLR() * scale;
}
function getStepsPerSecond() {
  const v = parseInt(document.getElementById('speedSlider').value);
  const cfg = {1:7.5, 2:15, 3:30, 4:60, 5:90, 6:120};
  return cfg[v] || 60;
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

function convergenceThreshold(path) {
  if (!path || path.length === 0) return 0;
  return Math.max(0.02, path[0].l * 0.03);
}

function convergenceStep(path) {
  if (!path || path.length === 0) return -1;
  const threshold = convergenceThreshold(path);
  return path.findIndex(p => p.l <= threshold);
}

function getInsight(s, ap) {
  const tm={sgd:'togSGD',mom:'togMOM',adam:'togADAM'};
  const active=['sgd','mom','adam'].filter(k=>document.getElementById(tm[k]).classList.contains('checked')&&ap[k]);
  if (!isRunning&&s===0) return `Ready: <strong>${activeLandscape().label}</strong> from <strong>${startPos.label}</strong>.`;
  if (s<5) return 'Early steps: paths follow the local gradient.';
  if (s<20) {
    if (active.includes('mom')) return '<strong>Momentum</strong> builds velocity along consistent slopes.';
    return 'Early phase: compare how each path bends.';
  }
  if (s<50) {
    if (active.includes('adam')&&active.includes('sgd')) {
      const al=ap.adam[s].l, sl=ap.sgd[s].l;
      if (al<sl*0.6) return '<strong>Adam</strong> is adapting step sizes fastest here.';
    }
    if (active.includes('mom')&&active.includes('sgd')) {
      const ml=ap.mom[s].l, sl=ap.sgd[s].l;
      if (ml<sl*0.7) return '<strong>Momentum</strong> is carrying through flatter regions.';
    }
    return 'Mid-run: steeper loss curves mean faster descent.';
  }
  const converged = active.map(k => ({ k, step: convergenceStep(ap[k]) })).filter(c => c.step >= 0 && c.step <= s);
  if (converged.length === active.length && active.length > 0) {
    return 'All visible optimizers have crossed the threshold.';
  }
  if (s<90) return 'Oscillation usually means α or β is too high.';
  const finals=active.map(k=>({k,l:ap[k][s].l})).sort((a,b)=>a.l-b.l);
  if (finals.length>0) {
    const w=finals[0];
    const fastest = active.map(k => ({ k, step: convergenceStep(ap[k]) })).filter(c => c.step >= 0).sort((a,b)=>a.step-b.step)[0];
    if (fastest) {
      return `<strong>${NAMES[fastest.k]}</strong> crossed threshold first. Best now: <strong>${NAMES[w.k]}</strong>.`;
    }
    return `<strong>${NAMES[w.k]}</strong> has the lowest current loss.`;
  }
  return 'Run complete. Adjust settings and compare again.';
}

// ── Sim loop ──────────────────────────────────────────────────────────────────
function tick(now) {
  if (!lastTickTime) lastTickTime = now;
  const dt = Math.min(0.1, (now - lastTickTime) / 1000);
  lastTickTime = now;

  animProgress = Math.min(animProgress + dt * getStepsPerSecond(), STEPS);
  step = Math.min(STEPS, Math.floor(animProgress));
  updateTrajectoryVisuals(animProgress);

  if (step !== lastUiStep) {
    lastUiStep = step;
    draw2D(paths, animProgress);
    updateLossReadouts(step);
    document.getElementById('stepLabel').textContent = step;
    document.getElementById('stepCounter').textContent = `step ${step} / ${STEPS}`;
    document.getElementById('progFill').style.width = (animProgress/STEPS*100).toFixed(1)+'%';
    document.getElementById('insightBox').innerHTML = getInsight(step, paths);
  }

  if (animProgress >= STEPS) {
    step = STEPS;
    isRunning = false;
    lastTickTime = 0;
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
  isRunning=false; step=0; animProgress=0; paths={}; lastTickTime=0; lastUiStep=-1;
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
    cancelAnimationFrame(animFrame); isRunning=false; lastTickTime=0;
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
    step=0; animProgress=0; lastUiStep=-1;
    redraw3D(paths, animProgress);
  }
  isRunning=true;
  lastTickTime=0;
  document.getElementById('runBtn').textContent='⏸  PAUSE';
  document.getElementById('runBtn').classList.add('running');
  animFrame=requestAnimationFrame(tick);
});

function updateSliderLabels() {
  document.getElementById('lrVal').textContent=getLR().toFixed(2);
  document.getElementById('momVal').textContent=getBeta().toFixed(2);
  const v=parseInt(document.getElementById('speedSlider').value);
  const labels={1:'0.1×',2:'0.25×',3:'0.5×',4:'1×',5:'1.5×',6:'2×'};
  document.getElementById('speedVal').textContent=labels[v]||v+'×';
}

function applyLandscapeDefaults() {
  const defaults = activeLandscape().defaults;
  document.getElementById('lrSlider').value = defaults.lr;
  document.getElementById('momSlider').value = defaults.beta;
  updateSliderLabels();
}

function renderLandscapeControls() {
  const select = document.getElementById('landscapeSelect');
  select.innerHTML = '';
  Object.entries(LANDSCAPES).forEach(([key, landscape]) => {
    const option = document.createElement('option');
    option.value = key;
    option.textContent = landscape.label;
    select.appendChild(option);
  });
  select.value = activeLandscapeKey;
}

function renderPresetControls() {
  const select = document.getElementById('presetSelect');
  select.innerHTML = '';
  Object.entries(activeLandscape().starts).forEach(([key, preset]) => {
    const option = document.createElement('option');
    option.value = key;
    option.textContent = preset.label;
    select.appendChild(option);
  });
  select.value = startKey;
}

document.getElementById('landscapeSelect').addEventListener('change', e => {
  activeLandscapeKey = e.target.value;
  LOSS_MAX = activeLandscape().lossMax;
  startKey = 'A';
  startPos = {...activeLandscape().starts[startKey]};
  applyLandscapeDefaults();
  renderLandscapeControls();
  renderPresetControls();
  buildSurface();
  resetAll();
});

document.getElementById('presetSelect').addEventListener('change', e => {
  startKey = e.target.value;
  startPos = {...activeLandscape().starts[startKey]};
  renderPresetControls();
  resetAll();
});

document.getElementById('lrSlider').addEventListener('input',()=>{
  updateSliderLabels();
});
document.getElementById('momSlider').addEventListener('input',()=>{
  updateSliderLabels();
});
document.getElementById('speedSlider').addEventListener('input',()=>{
  updateSliderLabels();
});

['togSGD','togMOM','togADAM'].forEach(id=>{
  document.getElementById(id).addEventListener('click', e => {
    e.preventDefault(); // prevent label from re-triggering the hidden checkbox
    const tog = document.getElementById(id);
    tog.classList.toggle('checked');
    if (Object.keys(paths).length) {
      redraw3D(paths, animProgress);
      draw2D(paths, animProgress);
      updateLossReadouts(step);
    }
  });
});

function resizeAll() {
  const w=container.clientWidth, h=container.clientHeight;
  renderer.setSize(w,h);
  camera.aspect=w/h; camera.updateProjectionMatrix();
  c2d.width=c2d.parentElement.clientWidth-28; c2d.height=80;
  draw2D(paths,step);
}

window.addEventListener('resize',()=>{ resizeAll(); redraw3D(paths,animProgress); });

// ── Init ──────────────────────────────────────────────────────────────────────
renderLandscapeControls();
renderPresetControls();
applyLandscapeDefaults();
applyOrbit();
buildSurface();
resizeAll();
redraw3D({},0);
renderLoop();

setTimeout(()=>{ document.getElementById('dragHint').style.opacity='0'; }, 5000);
