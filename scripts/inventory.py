import json,pathlib,math
root=pathlib.Path(__file__).resolve().parents[1];lock=json.loads((root/'MODEL_LOCK.json').read_text());rows=[]
for file,data in lock['tensor_inventory'].items():
 for name,t in data['tensors'].items():
  if name=='__metadata__':continue
  rows.append({'file':file,'name':name,'shape':t['shape'],'dtype':t['dtype'],'bytes':t['data_offsets'][1]-t['data_offsets'][0]})
base=[r for r in rows if r['file'].startswith('base/')];lang=[r for r in base if r['name'].startswith('model.language_model.')];vision=[r for r in base if r['name'].startswith('model.visual.')];other=[r for r in base if r not in lang+vision]
summary={'base_files_bytes':sum(d['file_bytes'] for f,d in lock['tensor_inventory'].items() if f.startswith('base/')),'base_tensor_bytes':sum(r['bytes'] for r in base),'language_tensor_bytes':sum(r['bytes'] for r in lang),'vision_tensor_bytes':sum(r['bytes'] for r in vision),'other_tensor_bytes':sum(r['bytes'] for r in other),'largest_tensor':max(base,key=lambda r:r['bytes']),'largest_language_tensor':max(lang,key=lambda r:r['bytes']),'unmerged_adapter_bytes':lock['tensor_inventory']['adapter/adapter_model.safetensors']['file_bytes'],'readout_bytes':lock['tensor_inventory']['adapter/decision_readout.safetensors']['file_bytes'],'language_parameters':sum(math.prod(r['shape']) for r in lang),'layer_types':json.loads((root/'checkpoints/base/config.json').read_text())['text_config']['layer_types']}
(root/'docs/capacity.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
