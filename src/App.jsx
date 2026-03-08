import { useState, useRef, useEffect, useMemo, useCallback } from "react";

/* ═══════════════════════════════════════════════════════
   SE(2) Differential-Drive Odometry with Uncertainty
   Based on Siegwart & Nourbakhsh, Ch.5 §5.2.4
   DGIST RT604 SLAM Course — Lab Exercise
   ═══════════════════════════════════════════════════════ */

const C = {
  bg: "#0a0f1a", panel: "#111827", bd: "#1e293b",
  grid: "#1a2332", gAxis: "#2a3a52",
  gt: "#00d4aa", pred: "#4da6ff", ellipse: "#ff5ca0",
  org: "#ff8c42", yel: "#fbbf24", pnk: "#ff5ca0",
  txt: "#e2e8f0", dim: "#64748b", mut: "#475569",
  mBg: "#0d1520", mBd: "#1e3a5f",
  formulaBg: "#0f1729", formulaBd: "#1e3050",
  thWedge: "#ff8c42", // orange for θ wedge
};

const rtd = r => r * 180 / Math.PI;

/* ── Seeded PRNG ── */
function makeRng(seed) {
  let _s = seed;
  const next = () => { _s = (_s * 16807 + 0) % 2147483647; return _s / 2147483647; };
  let spare = null;
  const randn = () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u, v, s;
    do { u = next() * 2 - 1; v = next() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
    const mul = Math.sqrt(-2.0 * Math.log(s) / s);
    spare = v * mul; return u * mul;
  };
  return { randn };
}

/* ── Odometry increment (Eq. 5.2-5.5) ── */
function odomIncrement(dsr, dsl, b) {
  const ds = (dsr + dsl) / 2, dth = (dsr - dsl) / b;
  return { dx: ds * Math.cos(dth / 2), dy: ds * Math.sin(dth / 2), dth, ds };
}
function wheelDistances(phiR, phiL, dt, r) { return { dsr: r * phiR * dt, dsl: r * phiL * dt }; }

/* ── Jacobians (Eq. 5.10, 5.11) ── */
function jacobianFp(ds, th, dth) {
  return [[1, 0, -ds * Math.sin(th + dth / 2)], [0, 1, ds * Math.cos(th + dth / 2)], [0, 0, 1]];
}
function jacobianFdelta(ds, th, dth, b) {
  const c = Math.cos(th + dth / 2), s = Math.sin(th + dth / 2);
  return [[.5*c - ds*s/(2*b), .5*c + ds*s/(2*b)], [.5*s + ds*c/(2*b), .5*s - ds*c/(2*b)], [1/b, -1/b]];
}

/* ── Matrix ops ── */
function mat3mul(A, B) { const r=[[0,0,0],[0,0,0],[0,0,0]]; for(let i=0;i<3;i++) for(let j=0;j<3;j++) for(let k=0;k<3;k++) r[i][j]+=A[i][k]*B[k][j]; return r; }
function mat3mulT(A, B) { const r=[[0,0,0],[0,0,0],[0,0,0]]; for(let i=0;i<3;i++) for(let j=0;j<3;j++) for(let k=0;k<3;k++) r[i][j]+=A[i][k]*B[j][k]; return r; }
function mat32diagT(F, d1, d2) { const r=[[0,0,0],[0,0,0],[0,0,0]]; for(let i=0;i<3;i++) for(let j=0;j<3;j++) r[i][j]=F[i][0]*d1*F[j][0]+F[i][1]*d2*F[j][1]; return r; }
function mat3add(A, B) { return A.map((row, i) => row.map((v, j) => v + B[i][j])); }

function eigen2x2(a, b, d) {
  const tr = a + d, det = a * d - b * b, disc = Math.max(0, tr * tr / 4 - det), sq = Math.sqrt(disc);
  return { l1: Math.max(0, tr/2 + sq), l2: Math.max(0, tr/2 - sq), angle: Math.abs(b) < 1e-12 ? (d > a ? Math.PI/2 : 0) : Math.atan2(tr/2 + sq - a, b) };
}

function det3x3(m) {
  return m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
}

/* ══════════════════════════════════════
   Simulation
   ══════════════════════════════════════ */
function simulate(steps, r, ell, kr, kl, seed) {
  const b = 2 * ell, rng = makeRng(seed);
  let gtPose = { x:0, y:0, th:0 }, noisyPose = { x:0, y:0, th:0 }, cov = [[0,0,0],[0,0,0],[0,0,0]];
  const gtTrail = [{ ...gtPose }], noisyTrail = [{ ...noisyPose, cov: cov.map(r=>[...r]) }];
  for (const step of steps) {
    const { dsr, dsl } = wheelDistances(step.pR, step.pL, step.dt, r);
    const gi = odomIncrement(dsr, dsl, b);
    gtPose = { x: gtPose.x + gi.dx*Math.cos(gtPose.th) - gi.dy*Math.sin(gtPose.th), y: gtPose.y + gi.dx*Math.sin(gtPose.th) + gi.dy*Math.cos(gtPose.th), th: gtPose.th + gi.dth };
    gtTrail.push({ ...gtPose });
    const nr = rng.randn()*Math.sqrt(kr*Math.abs(dsr)), nl = rng.randn()*Math.sqrt(kl*Math.abs(dsl));
    const ni = odomIncrement(dsr+nr, dsl+nl, b);
    const estTh = noisyPose.th; // save pre-update heading for Jacobian
    noisyPose = { x: noisyPose.x + ni.dx*Math.cos(noisyPose.th) - ni.dy*Math.sin(noisyPose.th), y: noisyPose.y + ni.dx*Math.sin(noisyPose.th) + ni.dy*Math.cos(noisyPose.th), th: noisyPose.th + ni.dth };
    const Fp = jacobianFp(gi.ds, estTh, gi.dth), Fd = jacobianFdelta(gi.ds, estTh, gi.dth, b);
    cov = mat3add(mat3mulT(mat3mul(Fp, cov), Fp), mat32diagT(Fd, kr*Math.abs(dsr), kl*Math.abs(dsl)));
    noisyTrail.push({ ...noisyPose, cov: cov.map(r=>[...r]) });
  }
  return { gtTrail, noisyTrail };
}

/* ══════════════════════════════════════
   Canvas Renderer
   ══════════════════════════════════════ */
function TraceCanvas({ gtTrail, noisyTrail, ellipseK, animIdx, sigma, showTheta, W = 560, H = 500 }) {
  const ref = useRef(null);
  const [hover, setHover] = useState(null);
  const allPts = useMemo(() => [...gtTrail, ...noisyTrail], [gtTrail, noisyTrail]);

  const handleMouseMove = useCallback((e) => {
    const cv = ref.current; if (!cv) return;
    const rect = cv.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (W / rect.width), my = (e.clientY - rect.top) * (H / rect.height);
    let mnX=1e9, mxX=-1e9, mnY=1e9, mxY=-1e9;
    for (const t of allPts) { mnX=Math.min(mnX,t.x); mxX=Math.max(mxX,t.x); mnY=Math.min(mnY,t.y); mxY=Math.max(mxY,t.y); }
    const pad=1.2; mnX-=pad; mxX+=pad; mnY-=pad; mxY+=pad;
    const rX=mxX-mnX||1, rY=mxY-mnY||1, sc=Math.min((W-60)/rX,(H-60)/rY);
    const ox=30+(W-60-rX*sc)/2, oy=30+(H-60-rY*sc)/2;
    const tcx=x=>ox+(x-mnX)*sc, tcy=y=>oy+(mxY-y)*sc;
    let bestD=400, bestI=-1;
    const n = animIdx >= 0 ? Math.min(animIdx+1, gtTrail.length) : gtTrail.length;
    for (let i=0; i<n; i++) { const d2=(tcx(gtTrail[i].x)-mx)**2+(tcy(gtTrail[i].y)-my)**2; if(d2<bestD){bestD=d2;bestI=i;} }
    setHover(bestI >= 0 ? bestI : null);
  }, [allPts, gtTrail, animIdx, W, H]);

  useEffect(() => {
    const cv = ref.current; if (!cv) return;
    const ctx = cv.getContext("2d");
    const dpr = window.devicePixelRatio||1;
    cv.width = W*dpr; cv.height = H*dpr; ctx.scale(dpr, dpr);

    const gtPts = animIdx >= 0 ? gtTrail.slice(0, animIdx+1) : gtTrail;
    const nPts = animIdx >= 0 ? noisyTrail.slice(0, animIdx+1) : noisyTrail;

    let mnX=1e9, mxX=-1e9, mnY=1e9, mxY=-1e9;
    for (const t of allPts) { mnX=Math.min(mnX,t.x); mxX=Math.max(mxX,t.x); mnY=Math.min(mnY,t.y); mxY=Math.max(mxY,t.y); }
    const pad=1.2; mnX-=pad; mxX+=pad; mnY-=pad; mxY+=pad;
    const rX=mxX-mnX||1, rY=mxY-mnY||1, sc=Math.min((W-60)/rX,(H-60)/rY);
    const ox=30+(W-60-rX*sc)/2, oy=30+(H-60-rY*sc)/2;
    const tc = (x,y) => [ox+(x-mnX)*sc, oy+(mxY-y)*sc];

    ctx.fillStyle = C.bg; ctx.fillRect(0,0,W,H);

    // Grid
    const maxRange=Math.max(rX,rY), rawStep=maxRange/8;
    const mag=Math.pow(10,Math.floor(Math.log10(rawStep)));
    const res=rawStep/mag;
    const niceStep = res<=1.5?mag:res<=3?2*mag:res<=7?5*mag:10*mag;
    ctx.font = "11px monospace";
    for (let gx=Math.ceil(mnX/niceStep)*niceStep; gx<=mxX; gx+=niceStep) {
      const [px]=tc(gx,0); ctx.strokeStyle=C.grid; ctx.lineWidth=0.5;
      ctx.beginPath(); ctx.moveTo(px,0); ctx.lineTo(px,H); ctx.stroke();
      ctx.fillStyle=C.dim; ctx.textAlign="center";
      ctx.fillText(Math.abs(gx)<1e-9?"0":niceStep>=1?gx.toFixed(0):gx.toFixed(1), px, H-4);
    }
    for (let gy=Math.ceil(mnY/niceStep)*niceStep; gy<=mxY; gy+=niceStep) {
      const [,py]=tc(0,gy); ctx.strokeStyle=C.grid; ctx.lineWidth=0.5;
      ctx.beginPath(); ctx.moveTo(0,py); ctx.lineTo(W,py); ctx.stroke();
      ctx.fillStyle=C.dim; ctx.textAlign="left";
      ctx.fillText(Math.abs(gy)<1e-9?"0":niceStep>=1?gy.toFixed(0):gy.toFixed(1), 3, py-3);
    }
    ctx.strokeStyle=C.gAxis; ctx.lineWidth=1;
    if(mnX<=0&&mxX>=0){const[px]=tc(0,0);ctx.beginPath();ctx.moveTo(px,0);ctx.lineTo(px,H);ctx.stroke();}
    if(mnY<=0&&mxY>=0){const[,py]=tc(0,0);ctx.beginPath();ctx.moveTo(0,py);ctx.lineTo(W,py);ctx.stroke();}

    // ── Covariance ellipses + θ wedge (all at every K steps) ──
    const K = Math.max(1, ellipseK);
    for (let i = K; i < nPts.length; i += K) {
      const pt = nPts[i] || noisyTrail[Math.min(i, noisyTrail.length-1)];
      const covM = noisyTrail[i]?.cov;
      if (!covM || !pt) continue;

      const [cx, cy] = tc(pt.x, pt.y);
      const { l1, l2, angle } = eigen2x2(covM[0][0], covM[0][1], covM[1][1]);
      const a = sigma * Math.sqrt(l1) * sc, bE = sigma * Math.sqrt(l2) * sc;
      const sigTh = Math.sqrt(Math.max(0, covM[2][2]));

      // xy covariance ellipse (pink)
      if (a >= 0.5 || bE >= 0.5) {
        ctx.save(); ctx.translate(cx, cy); ctx.rotate(-angle);
        ctx.beginPath(); ctx.ellipse(0, 0, Math.max(a, 0.5), Math.max(bE, 0.5), 0, 0, 2*Math.PI);
        ctx.fillStyle = C.ellipse + "18"; ctx.fill();
        ctx.strokeStyle = C.ellipse + "66"; ctx.lineWidth = 1.5; ctx.stroke();
        ctx.restore();
      }

      // θ uncertainty wedge (orange)
      if (showTheta && sigTh > 1e-6) {
        const wedgeR = Math.max(a, bE, 16);
        const thCenter = pt.th;
        const halfArc = sigma * sigTh;
        ctx.save(); ctx.translate(cx, cy);
        ctx.beginPath(); ctx.moveTo(0, 0);
        ctx.arc(0, 0, wedgeR, -(thCenter + halfArc), -(thCenter - halfArc));
        ctx.closePath();
        ctx.fillStyle = C.thWedge + "15"; ctx.fill();
        ctx.strokeStyle = C.thWedge + "44"; ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]); ctx.stroke(); ctx.setLineDash([]);
        ctx.restore();
      }
    }

    // Ghost trails if animating
    if (animIdx >= 0) {
      for (const [trail, color] of [[gtTrail, C.gt], [noisyTrail, C.pred]]) {
        if (trail.length < 2) continue;
        ctx.strokeStyle = color+"15"; ctx.lineWidth = 1; ctx.beginPath();
        const [sx,sy]=tc(trail[0].x,trail[0].y); ctx.moveTo(sx,sy);
        for(let i=1;i<trail.length;i++){const[tx,ty]=tc(trail[i].x,trail[i].y);ctx.lineTo(tx,ty);}
        ctx.stroke();
      }
    }

    // GT trail (green solid)
    if (gtPts.length > 1) {
      ctx.strokeStyle=C.gt; ctx.lineWidth=2.5; ctx.lineJoin="round"; ctx.lineCap="round"; ctx.setLineDash([]);
      ctx.beginPath(); const[sx,sy]=tc(gtPts[0].x,gtPts[0].y); ctx.moveTo(sx,sy);
      for(let i=1;i<gtPts.length;i++){const[tx,ty]=tc(gtPts[i].x,gtPts[i].y);ctx.lineTo(tx,ty);}
      ctx.stroke();
    }

    // Noisy trail (sky blue dashed)
    if (nPts.length > 1) {
      ctx.strokeStyle=C.pred; ctx.lineWidth=2; ctx.setLineDash([6,3]); ctx.lineJoin="round"; ctx.lineCap="round";
      ctx.beginPath(); const[sx,sy]=tc(nPts[0].x,nPts[0].y); ctx.moveTo(sx,sy);
      for(let i=1;i<nPts.length;i++){const[tx,ty]=tc(nPts[i].x,nPts[i].y);ctx.lineTo(tx,ty);}
      ctx.stroke(); ctx.setLineDash([]);
    }

    // Start marker
    {const[sx,sy]=tc(0,0);ctx.fillStyle=C.org;ctx.beginPath();ctx.arc(sx,sy,5,0,2*Math.PI);ctx.fill();}

    // Robot arrows
    const drawArrow = (pose, color, sz) => {
      const[cx2,cy2]=tc(pose.x,pose.y);
      ctx.save();ctx.translate(cx2,cy2);ctx.rotate(-pose.th);
      ctx.fillStyle=color+"55";ctx.strokeStyle=color;ctx.lineWidth=2;
      ctx.beginPath();ctx.moveTo(sz,0);ctx.lineTo(-sz*.7,-sz*.65);ctx.lineTo(-sz*.7,sz*.65);ctx.closePath();
      ctx.fill();ctx.stroke();
      ctx.fillStyle=C.yel;ctx.fillRect(-sz*.4,-sz*.8,sz*.5,3);ctx.fillRect(-sz*.4,sz*.8-3,sz*.5,3);
      ctx.restore();
    };
    if(gtPts.length>1)drawArrow(gtPts[gtPts.length-1],C.gt,13);
    if(nPts.length>1)drawArrow(nPts[nPts.length-1],C.pred,10);

    // Info overlay
    ctx.font="12px monospace";ctx.textAlign="right";ctx.setLineDash([]);
    const gf=gtPts[gtPts.length-1]||{x:0,y:0,th:0}, nf=nPts[nPts.length-1]||{x:0,y:0,th:0};
    ctx.fillStyle=C.dim;ctx.fillText(`step ${gtPts.length-1}/${gtTrail.length-1}`,W-8,H-8);
    ctx.fillStyle=C.gt;ctx.fillText(`GT  x=${gf.x.toFixed(2)} y=${gf.y.toFixed(2)} θ=${rtd(gf.th).toFixed(1)}°`,W-8,18);
    ctx.fillStyle=C.pred;ctx.fillText(`Est x=${nf.x.toFixed(2)} y=${nf.y.toFixed(2)} θ=${rtd(nf.th).toFixed(1)}°`,W-8,34);

    // Hover tooltip
    if (hover !== null && hover < gtPts.length && hover < nPts.length) {
      const gp=gtPts[hover], np=nPts[hover];
      const[hx,hy]=tc(gp.x,gp.y);
      ctx.strokeStyle=C.yel+"55";ctx.lineWidth=1;ctx.setLineDash([3,3]);
      ctx.beginPath();ctx.moveTo(hx,0);ctx.lineTo(hx,H);ctx.stroke();
      ctx.beginPath();ctx.moveTo(0,hy);ctx.lineTo(W,hy);ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle=C.yel;ctx.beginPath();ctx.arc(hx,hy,5,0,2*Math.PI);ctx.fill();
      const covH = noisyTrail[hover]?.cov;
      const detH = covH ? det3x3(covH) : 0;
      const sigThH = covH ? Math.sqrt(Math.max(0, covH[2][2])) : 0;
      const lines = [
        `step ${hover}`,
        `GT  (${gp.x.toFixed(2)}, ${gp.y.toFixed(2)}, ${rtd(gp.th).toFixed(1)}°)`,
        `Est (${np.x.toFixed(2)}, ${np.y.toFixed(2)}, ${rtd(np.th).toFixed(1)}°)`,
        `pos err ${Math.hypot(gp.x-np.x, gp.y-np.y).toFixed(3)} m`,
        `det(Σ₃ₓ₃) ${detH.toExponential(2)}`,
        `σ_θ ${rtd(sigThH).toFixed(2)}°`,
      ];
      ctx.font="bold 11px monospace";
      const tw=Math.max(...lines.map(l=>ctx.measureText(l).width))+16, th=lines.length*15+8;
      let tx=hx+14, ty=hy-th-8;
      if(tx+tw>W-4)tx=hx-tw-14; if(ty<4)ty=hy+14;
      ctx.fillStyle="#000000cc";ctx.beginPath();ctx.roundRect(tx+2,ty+2,tw,th,6);ctx.fill();
      ctx.fillStyle=C.panel+"ee";ctx.beginPath();ctx.roundRect(tx,ty,tw,th,6);ctx.fill();
      ctx.strokeStyle=C.yel+"66";ctx.lineWidth=1;ctx.beginPath();ctx.roundRect(tx,ty,tw,th,6);ctx.stroke();
      ctx.textAlign="left";
      const cols=[C.yel,C.gt,C.pred,C.pnk,C.yel,C.thWedge];
      lines.forEach((l,i)=>{ctx.fillStyle=cols[i];ctx.font=i===0?"bold 11px monospace":"11px monospace";ctx.fillText(l,tx+8,ty+15+i*15);});
    }
  }, [gtTrail, noisyTrail, allPts, animIdx, ellipseK, sigma, showTheta, hover, W, H]);

  return (
    <canvas ref={ref} onMouseMove={handleMouseMove} onMouseLeave={()=>setHover(null)}
      style={{width:W,height:H,borderRadius:12,border:`1px solid ${C.bd}`,cursor:hover!==null?"crosshair":"default"}} />
  );
}

/* ── UI helpers ── */
const Btn=({onClick,disabled,children,color=C.gt,style={}})=>(
  <button onClick={onClick} disabled={disabled} style={{
    background:disabled?C.bd:color+"22",color:disabled?C.mut:color,
    border:`1px solid ${disabled?C.bd:color+"55"}`,borderRadius:8,padding:"6px 14px",
    cursor:disabled?"not-allowed":"pointer",fontSize:12,fontWeight:600,fontFamily:"monospace",...style
  }}>{children}</button>
);
const Slider=({label,value,onChange,min,max,step,unit="",color=C.gt})=>(
  <div style={{marginBottom:8}}>
    <div style={{display:"flex",justifyContent:"space-between",fontSize:12,fontFamily:"monospace",color:C.dim,marginBottom:2}}>
      <span>{label}</span>
      <span style={{color}}>{typeof value==="number"?(step<0.01?value.toFixed(4):step<1?value.toFixed(2):value):value}{unit}</span>
    </div>
    <input type="range" min={min} max={max} step={step} value={value} onChange={e=>onChange(Number(e.target.value))}
      style={{width:"100%",accentColor:color}} />
  </div>
);

/* ══════════════════════════════════════
   Presets (tuned for default r=0.05, b=2*l=1.0)
   ══════════════════════════════════════ */
const PRESETS = {
  "Straight Line": Array(80).fill(null).map(()=>({pR:10,pL:10,dt:0.2})),
  "Circle": Array(120).fill(null).map(()=>({pR:13,pL:3,dt:0.15})),
  "Square": (()=>{
    // Turn: Δθ/step = r*(pR-pL)*dt/b = 0.05*20*dt/1.0 = dt
    // 10 steps * dt = π/2 → dt = π/20
    const turnDt=Math.PI/20, s=[];
    for(let side=0;side<4;side++){
      for(let i=0;i<25;i++) s.push({pR:10,pL:10,dt:0.2});
      for(let i=0;i<10;i++) s.push({pR:10,pL:-10,dt:turnDt});
    }
    return s;
  })(),
  "Figure-8": [
    ...Array(60).fill(null).map(()=>({pR:13,pL:3,dt:0.15})),
    ...Array(60).fill(null).map(()=>({pR:3,pL:13,dt:0.15})),
    ...Array(60).fill(null).map(()=>({pR:13,pL:3,dt:0.15})),
    ...Array(60).fill(null).map(()=>({pR:3,pL:13,dt:0.15})),
  ],
  "Zigzag": (()=>{
    const turnDt=Math.PI/20, s=[];
    for(let z=0;z<6;z++){
      for(let i=0;i<20;i++) s.push({pR:10,pL:10,dt:0.2});
      const dir=z%2===0?1:-1;
      for(let i=0;i<10;i++) s.push({pR:10*dir,pL:-10*dir,dt:turnDt});
    }
    return s;
  })(),
};

/* ══════════════════════════════════════
   Formulas Panel
   ══════════════════════════════════════ */
function FormulasPanel() {
  const S={fontFamily:"monospace",fontSize:12,lineHeight:1.9,color:C.dim};
  const eq={color:C.pred,fontWeight:600};
  const note={color:C.mut,fontSize:11};
  return (
    <div style={{background:C.formulaBg,border:`1px solid ${C.formulaBd}`,borderRadius:10,padding:14}}>
      <div style={{color:C.yel,fontWeight:700,fontSize:13,fontFamily:"monospace",marginBottom:8}}>
        Covariance Propagation (Siegwart, Eq. 5.8-5.11)
      </div>
      <div style={S}>
        <div style={{marginBottom:6}}><span style={{color:C.txt,fontWeight:600}}>Pose update</span> (Eq. 5.6-5.7):</div>
        <div style={eq}>{"  "}p' = p + [Δs·cos(θ + Δθ/2), Δs·sin(θ + Δθ/2), Δθ]ᵀ</div>
        <div style={{...note,marginBottom:8}}>{"  "}where Δs = (Δsᵣ+Δsₗ)/2, {"  "}Δθ = (Δsᵣ-Δsₗ)/b</div>
        <div style={{marginBottom:6}}><span style={{color:C.txt,fontWeight:600}}>Wheel noise model</span> (Eq. 5.8):</div>
        <div style={eq}>{"  "}Σ_Δ = diag(kᵣ|Δsᵣ|, kₗ|Δsₗ|)</div>
        <div style={{...note,marginBottom:8}}>{"  "}kᵣ, kₗ: error constants (wheel-floor interaction)</div>
        <div style={{marginBottom:6}}><span style={{color:C.txt,fontWeight:600}}>Covariance propagation</span> (Eq. 5.9):</div>
        <div style={{...eq,color:C.ellipse}}>{"  "}Σ_p' = F_p · Σ_p · F_pᵀ + F_Δ · Σ_Δ · F_Δᵀ</div>
        <div style={{marginTop:8,marginBottom:4}}><span style={{color:C.txt,fontWeight:600}}>Jacobian F_p</span> (Eq. 5.10):</div>
        <div style={{...eq,whiteSpace:"pre",fontSize:11}}>
{`  ┌ 1  0  -Δs·sin(θ+Δθ/2) ┐
  │ 0  1   Δs·cos(θ+Δθ/2) │
  └ 0  0         1         ┘`}
        </div>
        <div style={{marginTop:8,marginBottom:4}}><span style={{color:C.txt,fontWeight:600}}>Jacobian F_Δ</span> (Eq. 5.11):</div>
        <div style={{...eq,whiteSpace:"pre",fontSize:10}}>
{`  ┌ ½c - Δs·s/(2b)   ½c + Δs·s/(2b) ┐
  │ ½s + Δs·c/(2b)   ½s - Δs·c/(2b) │
  └      1/b              -1/b        ┘`}
        </div>
        <div style={{...note,marginTop:4}}>{"  "}c = cos(θ+Δθ/2), s = sin(θ+Δθ/2)</div>
        <div style={{marginTop:10,borderTop:`1px solid ${C.formulaBd}`,paddingTop:8}}>
          <span style={{color:C.txt,fontWeight:600}}>Visualization:</span>
          <div style={{...note,marginTop:4,lineHeight:1.7}}>
            <span style={{color:C.ellipse}}>Pink ellipse</span> = nσ of 2x2 xy-marginal of Σ₃ₓ₃.
            {" "}<span style={{color:C.thWedge}}>Orange wedge</span> = ±nσ_θ heading uncertainty.
            {" "}The xy-ellipse can appear to shrink when θ-xy cross-correlations change sign at heading reversals, but det(Σ₃ₓ₃) is monotonically non-decreasing.
          </div>
        </div>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════
   Main App
   ══════════════════════════════════════ */
export default function App() {
  const [mode, setMode] = useState("preset");
  const [r, setR] = useState(0.05);
  const [ell, setEll] = useState(0.5);
  const [kr, setKr] = useState(0.002);
  const [kl, setKl] = useState(0.002);
  const [ellipseK, setEllipseK] = useState(12);
  const [sigma, setSigma] = useState(3);
  const [seed, setSeed] = useState(42);
  const [preset, setPreset] = useState("Square");
  const [animIdx, setAnimIdx] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const animRef = useRef(null);
  const [showFormulas, setShowFormulas] = useState(true);
  const [showTheta, setShowTheta] = useState(true);

  const [iSteps, setISteps] = useState([]);
  const [iDriving, setIDriving] = useState(false);
  const [liveV, setLiveV] = useState(0);
  const [liveW, setLiveW] = useState(0);
  const keysRef = useRef(new Set());
  const LIVE_DT = 0.05, V_SPEED = 0.5, W_SPEED = 1.5;

  const twistToWheel = useCallback((v, w) => ({pR:(v+w*ell)/r, pL:(v-w*ell)/r}), [r, ell]);

  useEffect(() => {
    if (mode !== "interactive") return;
    const onDown = e => { if (!iDriving) return; if (["ArrowUp","ArrowDown","ArrowLeft","ArrowRight"].includes(e.key)){e.preventDefault();keysRef.current.add(e.key);} };
    const onUp = e => { keysRef.current.delete(e.key); };
    window.addEventListener("keydown", onDown); window.addEventListener("keyup", onUp);
    return () => { window.removeEventListener("keydown", onDown); window.removeEventListener("keyup", onUp); };
  }, [mode, iDriving]);

  useEffect(() => {
    if (mode !== "interactive" || !iDriving) { setLiveV(0); setLiveW(0); return; }
    const iv = setInterval(() => {
      const keys=keysRef.current; let v=0,w=0;
      if(keys.has("ArrowUp"))v+=V_SPEED; if(keys.has("ArrowDown"))v-=V_SPEED;
      if(keys.has("ArrowLeft"))w+=W_SPEED; if(keys.has("ArrowRight"))w-=W_SPEED;
      setLiveV(v); setLiveW(w);
      if(v!==0||w!==0){const wh=twistToWheel(v,w);setISteps(prev=>[...prev,{pR:wh.pR,pL:wh.pL,dt:LIVE_DT}]);}
    }, 50);
    return () => { clearInterval(iv); keysRef.current.clear(); setLiveV(0); setLiveW(0); };
  }, [mode, iDriving, twistToWheel]);

  const activeSteps = useMemo(() => mode==="interactive" ? iSteps : (PRESETS[preset]||[]), [mode, iSteps, preset]);
  const { gtTrail, noisyTrail } = useMemo(() => simulate(activeSteps, r, ell, kr, kl, seed), [activeSteps, r, ell, kr, kl, seed]);

  const finalErr = useMemo(() => {
    if(gtTrail.length<2) return 0;
    const g=gtTrail[gtTrail.length-1], n=noisyTrail[noisyTrail.length-1];
    return Math.hypot(g.x-n.x, g.y-n.y);
  }, [gtTrail, noisyTrail]);

  useEffect(() => {
    if(!playing){if(animRef.current)cancelAnimationFrame(animRef.current);return;}
    let idx=animIdx<0?0:animIdx;
    const speed=Math.max(1,Math.floor(gtTrail.length/300));
    const tick=()=>{idx+=speed;if(idx>=gtTrail.length){setPlaying(false);setAnimIdx(-1);return;}setAnimIdx(idx);animRef.current=requestAnimationFrame(tick);};
    animRef.current=requestAnimationFrame(tick);
    return()=>{if(animRef.current)cancelAnimationFrame(animRef.current);};
  }, [playing, gtTrail.length]);

  const resetAnim=()=>{setPlaying(false);setAnimIdx(-1);};
  const SS={section:{background:C.panel,borderRadius:10,padding:12,marginBottom:10,border:`1px solid ${C.bd}`}};

  return (
    <div style={{background:C.bg,minHeight:"100vh",padding:"20px 16px",color:C.txt}}>
      <div style={{maxWidth:1120,margin:"0 auto"}}>
        <div style={{textAlign:"center",marginBottom:16}}>
          <div style={{fontSize:11,color:C.gt,textTransform:"uppercase",letterSpacing:3,fontFamily:"monospace"}}>RT604 SLAM Course · Interactive Lab</div>
          <h1 style={{fontFamily:"monospace",fontSize:22,fontWeight:800,color:C.gt,margin:"4px 0",letterSpacing:1}}>
            SE(2) Odometry Uncertainty Propagation
          </h1>
          <div style={{fontFamily:"monospace",fontSize:12,color:C.dim,marginTop:4}}>
            <span style={{color:C.gt}}>GT</span>{" / "}
            <span style={{color:C.pred}}>Noisy Estimate</span>{" / "}
            <span style={{color:C.ellipse}}>Cov. Ellipse</span>{" / "}
            <span style={{color:C.thWedge}}>θ Wedge</span>
            {" "}(Siegwart & Nourbakhsh, §5.2.4)
          </div>
        </div>

        <div style={{display:"flex",justifyContent:"center",gap:6,marginBottom:14}}>
          {[{id:"preset",label:"Presets"},{id:"interactive",label:"⌨ Keyboard Drive"}].map(m=>(
            <Btn key={m.id} onClick={()=>{setMode(m.id);setIDriving(false);resetAnim();}}
              color={mode===m.id?C.gt:C.dim}
              style={{fontSize:13,padding:"8px 20px",fontWeight:700,background:mode===m.id?C.gt+"22":C.panel}}>
              {m.label}
            </Btn>
          ))}
        </div>

        {mode==="preset"&&(
          <div style={{display:"flex",justifyContent:"center",gap:6,marginBottom:14,flexWrap:"wrap"}}>
            {Object.keys(PRESETS).map(name=>(
              <Btn key={name} onClick={()=>{setPreset(name);resetAnim();}}
                color={preset===name?C.gt:C.org}
                style={{fontSize:12,fontWeight:700,padding:"6px 16px",background:preset===name?C.gt+"33":C.panel}}>
                {name} <span style={{fontSize:10,opacity:.6}}>({PRESETS[name].length})</span>
              </Btn>
            ))}
          </div>
        )}

        <div style={{display:"flex",gap:16,flexWrap:"wrap",justifyContent:"center"}}>
          <div>
            <TraceCanvas gtTrail={gtTrail} noisyTrail={noisyTrail} ellipseK={ellipseK}
              animIdx={animIdx} sigma={sigma} showTheta={showTheta} />

            <div style={{display:"flex",gap:6,marginTop:10,justifyContent:"center",flexWrap:"wrap"}}>
              {mode==="preset"&&(
                <>
                  <Btn onClick={()=>{if(playing)setPlaying(false);else{if(animIdx<0)setAnimIdx(0);setPlaying(true);}}}
                    color={playing?C.pnk:C.gt} style={{minWidth:100}}>
                    {playing?"⏸ Pause":"▶ Animate"}
                  </Btn>
                  <Btn onClick={resetAnim} color={C.dim}>Reset</Btn>
                </>
              )}
              <Btn onClick={()=>setSeed(s=>s+1)} color={C.yel}>Resample Noise</Btn>
            </div>
            {animIdx>=0&&mode==="preset"&&(
              <div style={{marginTop:8}}>
                <input type="range" min={0} max={gtTrail.length-1} value={animIdx}
                  onChange={e=>{setPlaying(false);setAnimIdx(Number(e.target.value));}}
                  style={{width:"100%",accentColor:C.gt}} />
              </div>
            )}

            <div style={{...SS.section,marginTop:10,display:"flex",gap:20,justifyContent:"center",flexWrap:"wrap"}}>
              <div style={{fontFamily:"monospace",fontSize:12}}>
                <span style={{color:C.dim}}>Steps: </span><span style={{color:C.txt,fontWeight:700}}>{activeSteps.length}</span>
              </div>
              <div style={{fontFamily:"monospace",fontSize:12}}>
                <span style={{color:C.dim}}>Final pos error: </span><span style={{color:C.pnk,fontWeight:700}}>{finalErr.toFixed(4)} m</span>
              </div>
              <div style={{fontFamily:"monospace",fontSize:12}}>
                <span style={{color:C.dim}}>Final θ error: </span>
                <span style={{color:C.pnk,fontWeight:700}}>
                  {(gtTrail.length>1?Math.abs(rtd(gtTrail[gtTrail.length-1].th-noisyTrail[noisyTrail.length-1].th)):0).toFixed(2)}°
                </span>
              </div>
            </div>
          </div>

          <div style={{width:340,flexShrink:0}}>
            {mode==="interactive"&&(
              <div style={{display:"flex",flexDirection:"column",gap:10,marginBottom:10}}>
                <div style={SS.section}>
                  <div style={{color:C.yel,fontWeight:700,fontSize:14,fontFamily:"monospace",marginBottom:8}}>⌨ Keyboard Drive Mode</div>
                  <div style={{color:C.dim,fontSize:12,fontFamily:"monospace",lineHeight:1.7,marginBottom:10}}>
                    Drive the robot with arrow keys. Watch uncertainty grow in real-time.
                  </div>
                  <div style={{display:"flex",justifyContent:"center",marginBottom:12}}>
                    <Btn onClick={()=>setIDriving(d=>!d)} color={iDriving?C.pnk:C.gt}
                      style={{fontSize:14,padding:"10px 28px",fontWeight:800,letterSpacing:1,
                        boxShadow:iDriving?`0 0 20px ${C.pnk}44`:`0 0 20px ${C.gt}44`}}>
                      {iDriving?"⏹ Stop Driving":"▶ Start Driving"}
                    </Btn>
                  </div>
                  {!iDriving&&iSteps.length===0&&(
                    <div style={{textAlign:"center",color:C.mut,fontSize:12,fontFamily:"monospace",marginBottom:8}}>Click "Start Driving" then use arrow keys.</div>
                  )}
                  {iDriving&&(
                    <div style={{textAlign:"center",color:C.gt,fontSize:12,fontFamily:"monospace",marginBottom:8}}>● DRIVING — press arrow keys now</div>
                  )}
                  <div style={{display:"flex",justifyContent:"center",marginBottom:12}}>
                    <div style={{display:"flex",flexDirection:"column",alignItems:"center",gap:4}}>
                      <div style={{width:48,height:40,borderRadius:8,display:"flex",alignItems:"center",justifyContent:"center",
                        fontFamily:"monospace",fontSize:14,fontWeight:700,
                        background:liveV>0?C.gt+"44":C.mBg,color:liveV>0?C.gt:C.dim,
                        border:`2px solid ${liveV>0?C.gt:C.mBd}`,transition:"all 0.1s"}}>↑</div>
                      <div style={{display:"flex",gap:4}}>
                        {[[liveW>0,C.yel,"←"],[liveV<0,C.pnk,"↓"],[liveW<0,C.yel,"→"]].map(([active,col,label],idx)=>(
                          <div key={idx} style={{width:48,height:40,borderRadius:8,display:"flex",alignItems:"center",justifyContent:"center",
                            fontFamily:"monospace",fontSize:14,fontWeight:700,
                            background:active?col+"44":C.mBg,color:active?col:C.dim,
                            border:`2px solid ${active?col:C.mBd}`,transition:"all 0.1s"}}>{label}</div>
                        ))}
                      </div>
                    </div>
                  </div>
                  <div style={{fontSize:12,fontFamily:"monospace",color:C.dim,lineHeight:1.8}}>
                    <div><span style={{color:C.gt}}>↑ / ↓</span> = forward / backward <span style={{color:C.txt}}>(v = ±{V_SPEED})</span></div>
                    <div><span style={{color:C.yel}}>← / →</span> = turn left / right <span style={{color:C.txt}}>(ω = ±{W_SPEED})</span></div>
                  </div>
                </div>
                <div style={{...SS.section,background:C.mBg,border:`1px solid ${C.mBd}`}}>
                  <div style={{color:C.dim,fontSize:11,fontFamily:"monospace",marginBottom:6}}>Live Command</div>
                  <div style={{display:"flex",gap:16,fontSize:14,fontFamily:"monospace",fontWeight:700}}>
                    <span style={{color:liveV!==0?C.gt:C.mut}}>v = {liveV.toFixed(1)} m/s</span>
                    <span style={{color:liveW!==0?C.yel:C.mut}}>ω = {liveW.toFixed(1)} rad/s</span>
                  </div>
                  <div style={{marginTop:8,display:"flex",gap:12,fontSize:12,fontFamily:"monospace",color:C.dim}}>
                    <span>Steps: <span style={{color:C.txt}}>{iSteps.length}</span></span>
                    <span>Time: <span style={{color:C.txt}}>{(iSteps.length*LIVE_DT).toFixed(1)}s</span></span>
                  </div>
                </div>
                <div style={{display:"flex",gap:6}}>
                  <Btn onClick={()=>{setISteps([]);resetAnim();}} color={C.pnk}>Clear Path</Btn>
                  <Btn onClick={()=>{setISteps(p=>p.slice(0,Math.max(0,p.length-20)));resetAnim();}}
                    color={C.dim} disabled={iSteps.length===0}>Undo (20 steps)</Btn>
                </div>
              </div>
            )}

            <div style={SS.section}>
              <div style={{color:C.gt,fontWeight:700,fontSize:13,fontFamily:"monospace",marginBottom:8}}>Robot Parameters</div>
              <Slider label="Wheel radius (r)" value={r} onChange={v=>{setR(v);resetAnim();}} min={0.02} max={0.3} step={0.01} unit=" m" color={C.gt} />
              <Slider label="Half-axle length (l)" value={ell} onChange={v=>{setEll(v);resetAnim();}} min={0.05} max={0.5} step={0.01} unit=" m" color={C.gt} />
            </div>

            <div style={{...SS.section,background:C.mBg,border:`1px solid ${C.mBd}`}}>
              <div style={{color:C.pnk,fontWeight:700,fontSize:13,fontFamily:"monospace",marginBottom:8}}>Noise Parameters</div>
              <Slider label="kᵣ (right wheel)" value={kr} onChange={v=>{setKr(v);resetAnim();}} min={0.0001} max={0.05} step={0.0001} unit="" color={C.pnk} />
              <Slider label="kₗ (left wheel)" value={kl} onChange={v=>{setKl(v);resetAnim();}} min={0.0001} max={0.05} step={0.0001} unit="" color={C.pnk} />
              <div style={{fontSize:11,fontFamily:"monospace",color:C.mut,marginTop:4}}>Σ_Δ = diag(kᵣ|Δsᵣ|, kₗ|Δsₗ|)</div>
            </div>

            <div style={SS.section}>
              <div style={{color:C.pred,fontWeight:700,fontSize:13,fontFamily:"monospace",marginBottom:8}}>Ellipse Display</div>
              <Slider label="Draw every K steps" value={ellipseK} onChange={setEllipseK} min={1} max={50} step={1} unit=" steps" color={C.pred} />
              <Slider label="Sigma multiplier" value={sigma} onChange={setSigma} min={1} max={5} step={0.5} unit="σ" color={C.pnk} />
              <div style={{marginTop:6,display:"flex",alignItems:"center",gap:8}}>
                <label style={{fontFamily:"monospace",fontSize:12,color:C.dim,display:"flex",alignItems:"center",gap:6,cursor:"pointer"}}>
                  <input type="checkbox" checked={showTheta} onChange={e=>setShowTheta(e.target.checked)} style={{accentColor:C.thWedge}} />
                  Show θ uncertainty wedge
                </label>
              </div>
            </div>

            <div style={{...SS.section,padding:10}}>
              <div style={{display:"flex",gap:14,fontSize:12,fontFamily:"monospace",flexWrap:"wrap",alignItems:"center"}}>
                <div style={{display:"flex",alignItems:"center",gap:6}}>
                  <div style={{width:20,height:3,background:C.gt,borderRadius:2}} />
                  <span style={{color:C.gt}}>Ground Truth</span>
                </div>
                <div style={{display:"flex",alignItems:"center",gap:6}}>
                  <div style={{width:20,height:0,borderTop:`2px dashed ${C.pred}`}} />
                  <span style={{color:C.pred}}>Noisy Odom</span>
                </div>
                <div style={{display:"flex",alignItems:"center",gap:6}}>
                  <div style={{width:14,height:10,border:`2px solid ${C.ellipse}66`,borderRadius:"50%",background:C.ellipse+"18"}} />
                  <span style={{color:C.ellipse}}>xy Cov</span>
                </div>
                <div style={{display:"flex",alignItems:"center",gap:6}}>
                  <div style={{width:0,height:0,borderLeft:"7px solid transparent",borderRight:"7px solid transparent",borderBottom:`10px solid ${C.thWedge}44`}} />
                  <span style={{color:C.thWedge}}>θ ±σ</span>
                </div>
              </div>
            </div>

            <Btn onClick={()=>setShowFormulas(v=>!v)} color={C.yel} style={{width:"100%",marginBottom:10}}>
              {showFormulas?"▲ Hide Formulas":"▼ Show Covariance Propagation"}
            </Btn>
            {showFormulas&&<FormulasPanel />}
          </div>
        </div>

        <div style={{textAlign:"center",marginTop:16,color:C.mut,fontSize:11,fontFamily:"monospace"}}>
          DGIST · Dept. of Robotics & Mechatronics · RT604 SLAM Course
        </div>
      </div>
    </div>
  );
}
