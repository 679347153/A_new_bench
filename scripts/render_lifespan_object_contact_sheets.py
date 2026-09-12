#!/usr/bin/env python3
"""Render 20 large contact sheets containing an overview and every object close-up."""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, tempfile
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCENES=(("00401","00401-H8rQCnvBgo6"),("00808","00808-y9hTuugGdiq"))

def label(img,s,xy,scale=.72,color=(245,245,245),thick=2):
 cv2.putText(img,str(s),xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,thick,cv2.LINE_AA)

def tile_image(src,w=960,h=540):
 out=np.full((h,w,3),(16,18,22),np.uint8)
 ratio=min(w/src.shape[1],(h-42)/src.shape[0])
 resized=cv2.resize(src,(int(src.shape[1]*ratio),int(src.shape[0]*ratio)))
 x=(w-resized.shape[1])//2; y=42+(h-42-resized.shape[0])//2
 out[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
 return out

def sheet(render_dir,layout,scene,index):
 payload=json.loads(layout.read_text())
 objects=payload.get('objects',[])
 overview=next(render_dir.glob('*_overview_*.png'))
 focus=sorted(render_dir.glob('*_focus_*.png'))
 entries=[('SCENE OVERVIEW',overview)]
 for n,(obj,path) in enumerate(zip(objects,focus),1):
  entries.append((f"{n:02d}  {obj.get('model_id',obj.get('name','object'))}  xyz={obj.get('position')}",path))
 cols=3; tw,th=1280,720; header=110
 rows=(len(entries)+cols-1)//cols
 canvas=np.full((header+rows*th,cols*tw,3),(25,27,31),np.uint8)
 time_label=payload.get('lifespan_generation',{}).get('time_label',layout.stem)
 label(canvas,f"HM3D {scene} | layout {index:02d} | {time_label} | every object close-up",(24,45),1.0,(255,255,255),2)
 label(canvas,f"Objects: {len(objects)}  |  original coordinates, initial-y-offset=0  |  Qwen-reviewed revision",(24,84),.65,(175,205,255),1)
 for i,(title,path) in enumerate(entries):
  r,c=divmod(i,cols); x,y=c*tw,header+r*th
  src=cv2.imread(str(path)); tile=tile_image(src,tw,th)
  canvas[y:y+th,x:x+tw]=tile
  cv2.rectangle(canvas,(x,y),(x+tw-1,y+th-1),(90,95,105),2)
  label(canvas,title,(x+12,y+29),.54,(255,235,165),1)
 return canvas

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True,type=Path)
 ap.add_argument('--python',default=sys.executable)
 ap.add_argument('--data-dir',type=Path,help='Optional HM3D root; otherwise use the project path resolver')
 ap.add_argument('--scene',choices=('00401','00808'),help='只重建指定场景')
 args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
 viewer=ROOT/'core/visualize_placed_layout.py'
 with tempfile.TemporaryDirectory(prefix='all_object_views_') as tmp:
  tmp=Path(tmp)
  for short,scene in SCENES:
   if args.scene and short != args.scene: continue
   outdir=args.output/short; outdir.mkdir(parents=True,exist_ok=True)
   layouts=sorted(
    path for path in (ROOT/'results/lifespan'/scene/'lifespan_qwen_10/grounded/layouts').glob('snapshot_*.json')
    if not path.stem.endswith('_offset_debug')
   )
   for index,layout in enumerate(layouts):
    payload=json.loads(layout.read_text()); count=len(payload.get('objects',[]))
    render=tmp/short/f'{index:02d}'
    cmd=[args.python,str(viewer),str(layout),'--scene',scene,'--headless','--headless-max-focus',str(count),
      '--initial-y-offset','0','--width','960','--height','540','--screenshot-dir',str(render)]
    if args.data_dir: cmd += ['--data-dir',str(args.data_dir)]
    subprocess.run(cmd,cwd=ROOT,check=True)
    focus=sorted(render.glob('*_focus_*.png'))
    if len(focus)!=count: raise RuntimeError(f'{scene} layout {index}: focus {len(focus)} != objects {count}')
    detail_dir=outdir/f'{short}_layout_{index:02d}_objects'
    detail_dir.mkdir(parents=True,exist_ok=True)
    overview=next(render.glob('*_overview_*.png'))
    shutil.copy2(overview,detail_dir/'00_scene_overview.png')
    for object_index,(obj,source) in enumerate(zip(payload.get('objects',[]),focus),1):
     model=str(obj.get('model_id',obj.get('name','object'))).replace('/','_')
     shutil.copy2(source,detail_dir/f'{object_index:02d}_{model}.png')
    image=sheet(render,layout,scene,index)
    dst=outdir/f'{short}_layout_{index:02d}_all_objects.png'
    cv2.imwrite(str(dst),image,[cv2.IMWRITE_PNG_COMPRESSION,3])
    print(f'[OK] {dst} objects={count} size={image.shape[1]}x{image.shape[0]}',flush=True)
 (args.output/'README.txt').write_text(
  '每个 HM3D 场景包含 10 张超大汇总图，并为每个布局建立 objects 文件夹。\n'
  '每个 objects 文件夹内含场景总览和该布局每个物体的一张 960x540 独立视角图；绿色轮廓表示目标物体。\n'
  '00401 每张含 17 个物体；00808 每张含 16 个物体。渲染使用原始坐标，未增加高度偏移。\n',encoding='utf-8')
if __name__=='__main__': main()
