# -*- coding: utf-8 -*-
"""Lightweight web editor for YOLO wing-detection labels.

Run:  .venv/bin/python tools/crop_web.py [--dir crop_yolo_v2/review] [--port 5005]
Open: http://127.0.0.1:5005

Edits the YOLO <stem>.txt files in place (normalized cx cy w h, class 0=wing).
Images are served downscaled for speed; labels are normalized so edits are
resolution-independent.
"""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, send_file, Response

app = Flask(__name__)
REVIEW = Path("crop_yolo_v2/review")
MAXW = 1500
_cache: dict[str, bytes] = {}

IMG_EXT = {".jpg", ".jpeg", ".png"}


def _stems():
    items = []
    for p in sorted(REVIEW.iterdir()):
        if p.suffix.lower() in IMG_EXT:
            lf = p.with_suffix(".txt")
            n = 0
            if lf.exists():
                n = sum(1 for ln in lf.read_text().splitlines() if ln.strip())
            items.append((p.name, n))
    # hardest (most boxes) first
    items.sort(key=lambda t: -t[1])
    return items


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.route("/api/list")
def api_list():
    return jsonify([{"name": n, "boxes": c} for n, c in _stems()])


@app.route("/img/<name>")
def img(name):
    if name in _cache:
        return send_file(io.BytesIO(_cache[name]), mimetype="image/jpeg")
    p = REVIEW / name
    real = os.path.realpath(p)
    im = cv2.imread(real)
    if im is None:
        return ("not found", 404)
    h, w = im.shape[:2]
    if w > MAXW:
        im = cv2.resize(im, (MAXW, int(h * MAXW / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])
    data = buf.tobytes()
    _cache[name] = data
    return send_file(io.BytesIO(data), mimetype="image/jpeg")


@app.route("/api/labels/<name>", methods=["GET", "POST"])
def labels(name):
    lf = (REVIEW / name).with_suffix(".txt")
    if request.method == "GET":
        boxes = []
        if lf.exists():
            for ln in lf.read_text().splitlines():
                parts = ln.split()
                if len(parts) >= 5:
                    _, cx, cy, w, h = parts[:5]
                    boxes.append([float(cx), float(cy), float(w), float(h)])
        return jsonify(boxes)
    data = request.get_json(force=True)
    lines = []
    for b in data.get("boxes", []):
        cx, cy, w, h = b
        cx = min(max(cx, 0), 1); cy = min(max(cy, 0), 1)
        w = min(max(w, 0), 1); h = min(max(h, 0), 1)
        if w < 0.002 or h < 0.002:
            continue
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    lf.write_text("\n".join(lines) + ("\n" if lines else ""))
    return jsonify({"ok": True, "saved": len(lines)})


HTML = r"""<!doctype html><html lang=ru><head><meta charset=utf-8>
<title>Разметка крыльев</title>
<style>
 *{box-sizing:border-box} body{margin:0;font:14px system-ui;background:#1e1e1e;color:#ddd;display:flex;height:100vh}
 #side{width:230px;background:#252526;overflow-y:auto;border-right:1px solid #333;flex:none}
 #side .it{padding:7px 10px;cursor:pointer;border-bottom:1px solid #2d2d2d;display:flex;justify-content:space-between}
 #side .it:hover{background:#2a2d2e} #side .it.act{background:#094771;color:#fff}
 #side .it .b{color:#888;font-size:12px} #side .it.done .b{color:#4ec9b0}
 #main{flex:1;display:flex;flex-direction:column;overflow:hidden}
 #bar{padding:8px 12px;background:#333;display:flex;gap:10px;align-items:center;flex:none}
 #bar b{color:#fff} #bar .sp{flex:1} button{background:#0e639c;color:#fff;border:0;padding:6px 12px;border-radius:4px;cursor:pointer}
 button:hover{background:#1177bb} #stage{flex:1;overflow:auto;position:relative;display:flex;align-items:flex-start;justify-content:center;padding:14px}
 #wrap{position:relative;line-height:0} #wrap img{max-width:100%;user-select:none;-webkit-user-drag:none}
 .box{position:absolute;border:2px solid #ff3b3b;background:rgba(255,59,59,.07);cursor:move}
 .box.sel{border-color:#21d4fd;background:rgba(33,212,253,.12);z-index:5}
 .h{position:absolute;width:11px;height:11px;background:#21d4fd;border:1px solid #003;border-radius:2px}
 .h.nw{left:-6px;top:-6px;cursor:nwse-resize}.h.ne{right:-6px;top:-6px;cursor:nesw-resize}
 .h.sw{left:-6px;bottom:-6px;cursor:nesw-resize}.h.se{right:-6px;bottom:-6px;cursor:nwse-resize}
 #hint{color:#aaa;font-size:12px}
</style></head><body>
<div id=side></div>
<div id=main>
 <div id=bar>
  <button onclick=prev()>◀ A</button><button onclick=next()>D ▶</button>
  <b id=title></b><span id=cnt></span><span class=sp></span>
  <span id=hint>тянуть на пустом — новый бокс · клик — выделить · Del — удалить · углы — растянуть · автосейв</span>
  <button onclick=save()>Сохранить ⌘S</button>
 </div>
 <div id=stage><div id=wrap><img id=img></div></div>
</div>
<script>
let list=[],idx=0,boxes=[],sel=-1,dirty=false;
const wrap=document.getElementById('wrap'),img=document.getElementById('img');
async function load(){list=await(await fetch('/api/list')).json();renderSide();if(list.length)open(0)}
function renderSide(){const s=document.getElementById('side');s.innerHTML='';list.forEach((it,i)=>{const d=document.createElement('div');d.className='it'+(i===idx?' act':'')+(it.done?' done':'');d.innerHTML=`<span>${i+1}. ${it.name.slice(0,14)}</span><span class=b>${it.boxes}</span>`;d.onclick=()=>open(i);s.appendChild(d)})}
async function open(i){if(dirty)await save();idx=i;sel=-1;
 img.src='/img/'+list[i].name;await img.decode().catch(()=>{});
 boxes=await(await fetch('/api/labels/'+list[i].name)).json();dirty=false;
 document.getElementById('title').textContent=`${i+1}/${list.length}  ${list[i].name}`;
 render();renderSide()}
function render(){[...wrap.querySelectorAll('.box')].forEach(e=>e.remove());
 const W=img.clientWidth,H=img.clientHeight;
 document.getElementById('cnt').textContent='  крыльев: '+boxes.length;
 boxes.forEach((b,i)=>{const[cx,cy,w,h]=b;const el=document.createElement('div');
  el.className='box'+(i===sel?' sel':'');
  el.style.left=(cx-w/2)*W+'px';el.style.top=(cy-h/2)*H+'px';el.style.width=w*W+'px';el.style.height=h*H+'px';
  el.dataset.i=i;
  if(i===sel)['nw','ne','sw','se'].forEach(c=>{const hd=document.createElement('div');hd.className='h '+c;hd.dataset.c=c;el.appendChild(hd)});
  wrap.appendChild(el)})}
function norm(e){const r=img.getBoundingClientRect();return[(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]}
let mode=null,start=null,orig=null,corner=null;
wrap.addEventListener('mousedown',e=>{
 const W=img.clientWidth,H=img.clientHeight;
 if(e.target.classList.contains('h')){mode='resize';corner=e.target.dataset.c;orig=[...boxes[sel]];start=norm(e);e.preventDefault();return}
 const bx=e.target.closest('.box');
 if(bx){sel=+bx.dataset.i;mode='move';orig=[...boxes[sel]];start=norm(e);render();e.preventDefault();return}
 // empty -> new box
 sel=-1;mode='new';start=norm(e);boxes.push([start[0],start[1],0,0]);sel=boxes.length-1;render();e.preventDefault()})
window.addEventListener('mousemove',e=>{if(!mode)return;const p=norm(e);
 if(mode==='new'){const[x0,y0]=start;boxes[sel]=[(x0+p[0])/2,(y0+p[1])/2,Math.abs(p[0]-x0),Math.abs(p[1]-y0)];render();dirty=true}
 else if(mode==='move'){const dx=p[0]-start[0],dy=p[1]-start[1];boxes[sel]=[orig[0]+dx,orig[1]+dy,orig[2],orig[3]];render();dirty=true}
 else if(mode==='resize'){let[cx,cy,w,h]=orig;let l=cx-w/2,t=cy-h/2,r=cx+w/2,b=cy+h/2;
  if(corner.includes('w'))l=p[0];if(corner.includes('e'))r=p[0];if(corner.includes('n'))t=p[1];if(corner.includes('s'))b=p[1];
  boxes[sel]=[(l+r)/2,(t+b)/2,Math.abs(r-l),Math.abs(b-t)];render();dirty=true}})
window.addEventListener('mouseup',()=>{if(mode==='new'&&(boxes[sel][2]<0.004||boxes[sel][3]<0.004)){boxes.pop();sel=-1;render()}mode=null})
window.addEventListener('keydown',e=>{
 if(e.key==='a'||e.key==='A')prev();else if(e.key==='d'||e.key==='D')next();
 else if(e.key==='Delete'||e.key==='Backspace'){if(sel>=0){boxes.splice(sel,1);sel=-1;dirty=true;render()}}
 else if((e.metaKey||e.ctrlKey)&&e.key==='s'){e.preventDefault();save()}})
function prev(){if(idx>0)open(idx-1)} function next(){if(idx<list.length-1)open(idx+1)}
async function save(){const r=await(await fetch('/api/labels/'+list[idx].name,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({boxes})})).json();
 dirty=false;list[idx].boxes=r.saved;list[idx].done=true;renderSide()}
window.addEventListener('resize',render);
load();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="crop_yolo_v2/review")
    ap.add_argument("--port", type=int, default=5005)
    a = ap.parse_args()
    global REVIEW
    REVIEW = Path(a.dir)
    print(f"Открой в браузере:  http://127.0.0.1:{a.port}")
    print(f"Папка меток: {REVIEW.resolve()}")
    app.run(host="127.0.0.1", port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
