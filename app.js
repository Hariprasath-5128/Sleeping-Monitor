import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

/**
 * SLEEPING MONITOR v3.0
 * Dual-Canvas Real-Time Monitoring System
 */

// --- Global State ---
const S = {
    bed: { w: 710, h: 918, x: 0, y: 0 },
    margins: { left: 122, right: 114, top: 80, bottom: 80 }, // Fixed clinical offsets
    person: { x: 0, y: 0, rot: 0, scale: 0.18 },
    limbs: { lArm: 30, rArm: 30 },
    status: 'SAFE', // SAFE, WARNING, DANGER, FALLEN
    mode: 'BODY',   // BODY, LIMB
    autoSim: false,
    drift: 0,
    fps: 0,
    parts: {
        head: 'safe', torso: 'safe',
        leftArm: 'safe', rightArm: 'safe',
        leftLeg: 'safe', rightLeg: 'safe'
    }
};

const THEME = {
    SAFE:    { primary: '#7dd3fc', secondary: '#10b981', glow: 'rgba(125, 211, 252, 0.1)', bg: '#f8fafc', emissive: 0x7dd3fc, text: '#1e293b' },
    WARNING: { primary: '#ea580c', secondary: '#ea580c', glow: 'rgba(234, 88, 12, 0.2)', bg: '#fff7ed', emissive: 0xea580c, text: '#9a3412' },
    DANGER:  { primary: '#dc2626', secondary: '#dc2626', glow: 'rgba(220, 38, 38, 0.3)', bg: '#fef2f2', emissive: 0xdc2626, text: '#991b1b' },
    FALLEN:  { primary: '#7f1d1d', secondary: '#7f1d1d', glow: 'rgba(127, 29, 29, 0.4)', bg: '#450a0a', emissive: 0xff0000, text: '#ffffff' }
};

// --- Config & UI Elements ---
const canvas2d = document.getElementById('simCanvas');
const ctx = canvas2d.getContext('2d');
const threeContainer = document.getElementById('threeOverlay');
const eventLog = document.getElementById('eventLog');
const statusBanner = document.getElementById('statusBanner');
const statusLabel = document.getElementById('statusLabel');
const fpsCounter = document.getElementById('fpsCounter');

// --- Three.js Globals ---
let scene, camera, renderer, humanModel, limbs = {};
let lastTime = 0;
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

// --- 1. Initialization ---
function init() {
    setupThree();
    setupEventListeners();
    loadModel();
    
    // Initial UI update
    updateUIVals();
    logEvent('System initialized. Loading 3D models...', 'info');
    
    requestAnimationFrame(loop);
}

function setupThree() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(35, window.innerWidth / window.innerHeight, 0.1, 5000);
    camera.position.set(0, 1000, 0); // Top-down
    camera.lookAt(0, 0, 0);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    threeContainer.appendChild(renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0x00d4ff, 1.2);
    mainLight.position.set(500, 1000, 200);
    scene.add(mainLight);
    
    const purpleLight = new THREE.PointLight(0x8b5cf6, 1.5, 2000);
    purpleLight.position.set(-400, 500, -200);
    scene.add(purpleLight);

    window.addEventListener('resize', onResize);
    onResize();
}

function onResize() {
    const w = canvas2d.clientWidth;
    const h = canvas2d.clientHeight;
    
    canvas2d.width = w;
    canvas2d.height = h;
    
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
}

// --- 2. Model Loading & Auto-Scaling ---
function loadModel() {
    console.log('Starting model load from models/human.obj');
    logEvent('Starting model load...', 'info');
    const loader = new OBJLoader();
    loader.load('models/human.obj', (obj) => {
        console.log('Model loaded successfully', obj);
        humanModel = new THREE.Group();
        scene.add(humanModel);

        const material = new THREE.MeshPhongMaterial({
            color: 0x00d4ff,
            transparent: true,
            opacity: 0.25,
            side: THREE.DoubleSide,
            emissive: 0x004466,
            emissiveIntensity: 0.5
        });

        const wireMaterial = new THREE.MeshBasicMaterial({
            color: 0x00d4ff,
            wireframe: true,
            transparent: true,
            opacity: 0.1
        });

        // Use a separate group for wires to avoid tree-traversal recursion
        const wireGroup = new THREE.Group();
        
        obj.traverse((child) => {
            if (child.isMesh) {
                child.material = material;
                const wire = child.clone();
                wire.material = wireMaterial;
                wireGroup.add(wire);
            }
        });

        // Center and orient the model
        const mainGroup = new THREE.Group();
        mainGroup.add(obj);
        mainGroup.add(wireGroup);
        
        // Initial rotation to lying position (assuming head is Y-up)
        mainGroup.rotation.x = -Math.PI / 2;
        humanModel.add(mainGroup);

        // --- Auto-Scaling & Centering Logic ---
        const bbox = new THREE.Box3().setFromObject(humanModel);
        const center = bbox.getCenter(new THREE.Vector3());
        const size = bbox.getSize(new THREE.Vector3());
        
        // Offset to center the geometry
        mainGroup.position.sub(center);
        
        // Scale to fit mattress (Zoomed out)
        const maxDim = Math.max(size.x, size.y, size.z);
        const targetDim = S.bed.h * 0.55; 
        const scaleFactor = targetDim / maxDim;
        
        S.person.scale = scaleFactor;
        humanModel.scale.set(scaleFactor, scaleFactor, scaleFactor);
        
        // Initial Material setup for clinical look (Reference Cyan)
        humanModel.traverse(c => {
            if(c.isMesh && c.material) {
                const mats = Array.isArray(c.material) ? c.material : [c.material];
                mats.forEach(m => {
                    m.color.set('#7dd3fc');
                    if (m.emissive) {
                        m.emissive.setHex(0x7dd3fc);
                        m.emissiveIntensity = 0.6;
                    }
                    m.opacity = 0.35;
                });
            }
        });
        
        logEvent('3D Human Model centered and loaded.', 'safe');
    }, undefined, (err) => {
        logEvent('Failed to load models/human.obj.', 'danger');
        console.error(err);
    });
}

// --- 3. Animation Loop ---
function loop(time) {
    const dt = (time - lastTime) / 1000;
    lastTime = time;
    
    update(dt);
    drawBase();
    
    if (renderer) renderer.render(scene, camera);
    
    if (time % 1000 < 20) fpsCounter.innerText = `${Math.round(1/dt)} FPS`;
    
    requestAnimationFrame(loop);
}

function update(dt) {
    // Handling Drift
    if (S.autoSim) {
        if (Math.abs(S.drift) < 0.1) S.drift = 15; // Initial push
        S.person.x += S.drift * dt;
        if (Math.abs(S.person.x) > 200) S.drift *= -1; // Bounce for demo
    }

    // Sync 3D Model Position
    if (humanModel) {
        humanModel.position.x = S.person.x;
        humanModel.position.z = S.person.y;
        
        // Breathing animation
        const breath = 1 + Math.sin(Date.now() * 0.002) * 0.015;
        humanModel.scale.set(S.person.scale, S.person.scale * breath, S.person.scale);
        
        // Rotation
        humanModel.rotation.y = THREE.MathUtils.lerp(humanModel.rotation.y, S.person.rot, 0.1);
    }

    checkStatus();
}

// --- 4. 2D Bed Drawing ---
function drawBase() {
    ctx.clearRect(0, 0, canvas2d.width, canvas2d.height);
    
    const cx = canvas2d.width / 2;
    const cy = canvas2d.height / 2;
    
    // 1. Grid Background
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.02)';
    ctx.lineWidth = 1;
    const step = 40;
    for(let i=0; i<canvas2d.width; i+=step) { ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, canvas2d.height); ctx.stroke(); }
    for(let i=0; i<canvas2d.height; i+=step) { ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(canvas2d.width, i); ctx.stroke(); }

    // 2. Bed Construction
    const bw = S.bed.w;
    const bh = S.bed.h;
    const theme = THEME[S.status] || THEME.SAFE;
    
    // Frame (Solid Brown per Reference)
    ctx.fillStyle = '#4d3d31';
    ctx.shadowBlur = 15;
    ctx.shadowColor = 'rgba(0,0,0,0.4)';
    drawRoundRect(ctx, cx - bw/2 - 20, cy - bh/2 - 20, bw + 40, bh + 40, 20);
    ctx.fill();
    ctx.shadowBlur = 0;

    // Mattress (Reference Cream)
    ctx.fillStyle = '#f5f0e6';
    drawRoundRect(ctx, cx - bw/2, cy - bh/2, bw, bh, 12);
    ctx.fill();
    
    // Safety Zones - Exact Vertical Strips with Fixed Head/Foot
    const sl = S.margins.left;
    const sr = S.margins.right;
    const st = S.margins.top;
    const sb = S.margins.bottom;
    
    // Light Green Safe Strip (Clinical Tint)
    ctx.fillStyle = 'rgba(16, 185, 129, 0.05)';
    ctx.fillRect(cx - bw/2 + sl, cy - bh/2 + st, bw - sl - sr, bh - st - sb);
    
    // Boundary Lines (Mint Green Dashed - Full Height)
    ctx.setLineDash([8, 8]);
    ctx.strokeStyle = '#86efac';
    ctx.lineWidth = 2;
    
    // Exact vertical lines from image
    ctx.beginPath();
    ctx.moveTo(cx - bw/2 + sl, cy - bh/2); ctx.lineTo(cx - bw/2 + sl, cy + bh/2);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(cx + bw/2 - sr, cy - bh/2); ctx.lineTo(cx + bw/2 - sr, cy + bh/2);
    ctx.stroke();
    
    ctx.setLineDash([]);
    
    // 3. Bed Labels
    ctx.font = '700 12px Inter';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    
    // External directional labels
    ctx.fillText('HEAD', cx, cy - bh/2 - 40);
    ctx.fillText('FOOT', cx, cy + bh/2 + 50);
    
    ctx.save();
    ctx.translate(cx - bw/2 - 50, cy);
    ctx.rotate(-Math.PI/2);
    ctx.fillText('LEFT', 0, 0);
    ctx.restore();
    
    ctx.save();
    ctx.translate(cx + bw/2 + 50, cy);
    ctx.rotate(Math.PI/2);
    ctx.fillText('RIGHT', 0, 0);
    ctx.restore();

    // Internal Status Labels (Exact Image Match)
    ctx.font = '800 11px Inter';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#86efac';
    ctx.fillText('SAFE ZONE', cx, cy - bh/2 + 25);

    // "WARN" in side gutters (Reference Orange)
    ctx.fillStyle = '#fb923c';
    if (sl > 30) ctx.fillText('WARN', cx - bw/2 + sl/2, cy);
    if (sr > 30) ctx.fillText('WARN', cx + bw/2 - sr/2, cy);

    // Pillow (Reference Placement)
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    drawRoundRect(ctx, cx - 80, cy - bh/2 + 35, 160, 90, 22);
    ctx.fill();
    ctx.stroke();
}

// --- 5. Status Logic ---
function checkStatus() {
    const px = S.person.x;
    const py = S.person.y;
    const bw = S.bed.w;
    const bh = S.bed.h;
    const sl = S.margins.left;
    const sr = S.margins.right;
    const st = S.margins.top;
    const sb = S.margins.bottom;
    
    // Boundaries
    const xMin = -bw/2 + sl;
    const xMax = bw/2 - sr;
    const yMin = -bh/2 + st;
    const yMax = bh/2 - sb;
    
    // DETECTION WIDTHS (Enhanced Sensitivity)
    // We increase these to ensure early warnings for limb-crossing.
    const torsoWidth = 80;     // Strict mass detection (Danger)
    const fullBodyWidth = 240;  // Wide limb detection (Warning)
    
    const torsoXMin = px - torsoWidth/2;
    const torsoXMax = px + torsoWidth/2;
    const bodyXMin = px - fullBodyWidth/2;
    const bodyXMax = px + fullBodyWidth/2;
    
    const warningBuffer = 15; // Trigger early warning when nearing boundary
    
    // Status Logic
    let newStatus = 'SAFE';

    // 1. DANGER: Torso crosses into the warning gutter
    if (torsoXMin < xMin || torsoXMax > xMax) {
        newStatus = 'DANGER';
    } 
    // 2. WARNING: Arms/Legs cross OR torso is within the proximity buffer
    else if (bodyXMin < xMin || bodyXMax > xMax || 
             torsoXMin < xMin + warningBuffer || torsoXMax > xMax - warningBuffer) {
        newStatus = 'WARNING';
    }
    
    // FALLEN: Center of mass is significantly outside the mattress
    if (Math.abs(px) > bw/2 + 20 || Math.abs(py) > bh/2 + 20) newStatus = 'FALLEN';
    
    if (newStatus !== S.status) {
        S.status = newStatus;
        updateStatusUI();
        logEvent(`System Status Change: ${newStatus}`, newStatus.toLowerCase());
    }
}

function updateStatusUI() {
    statusBanner.className = `status-${S.status.toLowerCase()}`;
    statusLabel.innerText = S.status;
    
    const theme = THEME[S.status] || THEME.SAFE;
    
    // Update 3D Model Materials (Handle Arrays & Emissive Checks)
    if (humanModel) {
        humanModel.traverse(child => {
            if (child.isMesh && child.material) {
                const mats = Array.isArray(child.material) ? child.material : [child.material];
                mats.forEach(m => {
                    if (m.emissive) m.emissive.setHex(theme.emissive);
                    if (m.color) m.color.set(theme.primary);
                });
            }
        });
    }

    // Update body part Analysis
    const indicators = document.querySelectorAll('.part-dot');
    indicators.forEach(dot => {
        dot.className = `part-dot ${S.status.toLowerCase()}`;
    });
}

// --- 6. Event Listeners ---
function setupEventListeners() {
    // Bed Customization
    document.getElementById('bedWidth').addEventListener('input', (e) => { S.bed.w = parseInt(e.target.value); updateUIVals(); });
    document.getElementById('bedHeight').addEventListener('input', (e) => { S.bed.h = parseInt(e.target.value); updateUIVals(); });
    
    // Margins
    document.getElementById('leftMargin').addEventListener('input', (e) => { S.margins.left = parseInt(e.target.value); updateUIVals(); });
    document.getElementById('rightMargin').addEventListener('input', (e) => { S.margins.right = parseInt(e.target.value); updateUIVals(); });

    // Mode Toggle
    document.getElementById('bodyModeBtn').addEventListener('click', () => { S.mode = 'BODY'; toggleModeBtns(); });
    document.getElementById('limbModeBtn').addEventListener('click', () => { S.mode = 'LIMB'; toggleModeBtns(); });

    // Sim Controls
    document.getElementById('autoSimBtn').addEventListener('click', (e) => {
        S.autoSim = !S.autoSim;
        e.target.innerText = S.autoSim ? 'Stop Demo' : 'Start Demo';
        e.target.classList.toggle('active', S.autoSim);
        logEvent(S.autoSim ? 'Auto Demo Started' : 'Auto Demo Stopped');
    });

    document.getElementById('resetBtn').addEventListener('click', () => {
        S.person.x = 0; S.person.y = 0; S.autoSim = false; S.drift = 0;
        document.getElementById('autoSimBtn').innerText = 'Start Demo';
        document.getElementById('autoSimBtn').classList.remove('active');
        logEvent('Simulation Reset');
    });

    document.getElementById('simLeftBtn').addEventListener('click', () => S.drift = -40);
    document.getElementById('simRightBtn').addEventListener('click', () => S.drift = 40);

    // Dragging Logic (Precision Hit Detection)
    let isDragging = false;
    canvas2d.addEventListener('mousedown', (e) => {
        if (!humanModel || S.autoSim) return;

        // Normalized Device Coordinates
        const rect = canvas2d.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObject(humanModel, true);

        if (intersects.length > 0) {
            isDragging = true;
            canvas2d.style.cursor = 'grabbing';
        }
    });

    window.addEventListener('mouseup', () => {
        isDragging = false;
        canvas2d.style.cursor = 'default';
    });

    window.addEventListener('mousemove', (e) => {
        // Update mouse position for hover checks
        const rect = canvas2d.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        // Visual feedback (Pointer on hover)
        if (!isDragging && humanModel && !S.autoSim) {
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(humanModel, true);
            canvas2d.style.cursor = (intersects.length > 0) ? 'pointer' : 'default';
        }

        if (!isDragging || S.autoSim) return;
        S.person.x += e.movementX;
        S.person.y += e.movementY;
        canvas2d.style.cursor = 'grabbing';
    });
}

function updateUIVals() {
    document.getElementById('bedWVal').innerText = S.bed.w;
    document.getElementById('bedHVal').innerText = S.bed.h;
    document.getElementById('leftVal').innerText = S.margins.left;
    document.getElementById('rightVal').innerText = S.margins.right;
}

function toggleModeBtns() {
    document.getElementById('bodyModeBtn').classList.toggle('active', S.mode === 'BODY');
    document.getElementById('limbModeBtn').classList.toggle('active', S.mode === 'LIMB');
}

// --- Utils ---
function logEvent(msg, type = 'info') {
    const time = new Date().toTimeString().split(' ')[0];
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `<span class="time">[${time}]</span><span class="msg">${msg}</span>`;
    eventLog.prepend(entry);
    if(eventLog.childNodes.length > 50) eventLog.lastChild.remove();
}

function drawRoundRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
}

// Run App
init();
