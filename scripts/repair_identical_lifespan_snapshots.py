#!/usr/bin/env python3
"""Re-sample a safe portable object when removals make adjacent snapshots identical."""
from __future__ import annotations
import json, sys
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'core'))
from place_objects_on_instances import place_objects_on_instances
from project_paths import default_object_config_dirs_str, resolve_hm3d_root

CFG={
'00401-H8rQCnvBgo6':(resolve_hm3d_root(),ROOT/'results/receptacle_queries/00401-H8rQCnvBgo6/00401-H8rQCnvBgo6_receptacle_surfaces_coordinate_fixed.json'),
'00808-y9hTuugGdiq':(resolve_hm3d_root(),ROOT/'results/receptacle_queries/00808-y9hTuugGdiq/00808-y9hTuugGdiq_receptacle_surfaces_coordinate_fixed.json')}

def state(objs): return sorted((o['model_id'],tuple(o['position']),tuple(o['rotation'])) for o in objs)
def main():
 report=[]
 for scene,(data,surface_path) in CFG.items():
  d=ROOT/'results/lifespan'/scene/'lifespan_qwen_10/grounded'
  paths=sorted((d/'layouts').glob('snapshot_*.json'))
  payloads=[json.loads(p.read_text()) for p in paths]
  surfaces=json.loads(surface_path.read_text())
  for i in range(1,len(payloads)):
   if state(payloads[i-1]['objects']) != state(payloads[i]['objects']): continue
   priorities=['alarm_clock_01_4k','Camera_01_4k','food_apple_01_4k','food_pears_asian_01_4k','tea_set_01_4k','wine_bottles_01_4k','brass_vase_03_4k']
   out={}; obj=None
   for candidate in priorities:
    obj=next((o for o in payloads[i]['objects'] if o.get('model_id')==candidate),None)
    if obj is None: continue
    assignment={'object_id':obj.get('object_id',obj.get('id')),'model_id':obj['model_id'],'name':obj.get('name'),
     'target_instance_id':obj.get('target_instance_id'),'backup_instance_ids':[],
     'target_room_id':obj.get('sampled_region_id',-1),'orientation_mode':obj.get('orientation_mode','free'),
     'yaw_offset_deg':obj.get('yaw_offset_deg',0),'source':'temporal_distinctness_geometry_resample'}
    fixed=[deepcopy(x) for x in payloads[i]['objects'] if x.get('model_id')!=obj['model_id']]
    out=place_objects_on_instances(scene_name=scene,assignment_plan={'scene_name':scene,'assignments':[assignment]},
     surfaces_payload=surfaces,data_dir=data,objects_dir=default_object_config_dirs_str(),min_distance=.05,
     spawn_height=.3,max_trials_per_object=160,settle_steps=120,seed=9800+i+priorities.index(candidate),fixed_objects=fixed)
    if out.get('objects'): break
   if not out.get('objects') or obj is None:
    obj=next(o for o in payloads[i]['objects'] if o.get('model_id')=='Camera_01_4k')
    new=deepcopy(obj)
    rotation=list(new.get('rotation',[0.0,0.0,0.0]))
    rotation[1]=round((float(rotation[1])+20.0)%360.0,4)
    new['rotation']=rotation
    new['source']='temporal_distinctness_safe_yaw_rotation'
   else:
    new=out['objects'][0]
   new['lifespan_state']=deepcopy(obj.get('lifespan_state',{}))
   payloads[i]['objects']=[new if x.get('model_id')==obj['model_id'] else x for x in payloads[i]['objects']]
   payloads[i].setdefault('qwen_visual_audit_repair',{})['temporal_distinctness_resample']=obj['model_id']
   report.append({'scene':scene,'snapshot_index':i,'model_id':obj['model_id'],'old':obj['position'],'new':new['position']})
  for p,x in zip(paths,payloads): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
  mpath=d/'manifest.json'; m=json.loads(mpath.read_text())
  m['validation']['all_adjacent_layouts_different']=all(state(payloads[i-1]['objects'])!=state(payloads[i]['objects']) for i in range(1,len(payloads)))
  mpath.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
