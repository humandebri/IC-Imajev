#!/usr/bin/env python3
"""Freeze a short-input subset without inspecting predictions or gold labels."""
import collections,hashlib,json,pathlib,subprocess,sys
from prepare_text import TextPreparer,shorten_header
from vision_decision.contracts import ChoiceField,Option
from vision_decision.scoring import compile_question
import evaluate_prompt_accuracy as e
R=e.ROOT;D=R/'artifacts/decision-index-short-v1';KIT=R/'artifacts/decision-index-v1/source/decision-index'
sys.path.insert(0,str(KIT))
from decision_index.suite.build.freeze import gid

def main():
    D.mkdir(parents=True,exist_ok=True)
    if (D/'inputs.json').exists():raise RuntimeError('Refusing to replace a frozen sample')
    normal=R/'artifacts/decision-index-v1/work/artifacts/benchmark-suite/normalized'
    p=TextPreparer();records=[];selected=[];inventory={};seed=20261005
    families=['WinoGrande','HellaSwag','Humicroedit','iSarcasmEval-A-En','iSarcasmEval-C-En']
    source_paths=[normal/f'{family}.jsonl' for family in families]
    exclusions=set(json.loads((KIT/'hub/excluded-questions.json').read_text())['rows'])
    for family,path in zip(families,source_paths):
        rows=[json.loads(line) for line in path.open()]
        rows.sort(key=lambda row:hashlib.sha256(f'{seed}:{family}:{gid(row)}'.encode()).hexdigest())
        rejected=collections.Counter();scanned=0;count=0
        for row in rows:
            scanned+=1
            run_id=f'{40 if family.startswith("iSarcasm") else {"WinoGrande":28,"HellaSwag":29,"Humicroedit":21}[family]}:{family}:{row["id"]}'
            if run_id in exclusions:rejected['release_exclusion']+=1;continue
            assert len(row['questions'])==1
            key,q=next(iter(row['questions'].items()));criteria=q['criteria']
            assert q['type']=='choice' and 2<=len(criteria)<=7
            options=[dict(value=k,description=v) for k,v in criteria.items()]
            field=ChoiceField(id='eligibility',type='choice',question=q['instructions'],options=[Option(**x) for x in options])
            header,_,texts=compile_question(field,row['state'],'standard');header=shorten_header(header);texts[-1]='unknown'
            labels=[b['code'] for b in p.binding['codes'][:len(texts)]]
            prompt=header+'\n'.join(f'{label}: {text}' for label,text in zip(labels,texts))
            ids=p.tokenizer.encode(p.render(prompt),add_special_tokens=False)
            prefix=27 if isinstance(row['state'],str) and row['state'] else 26
            if not 1<=len(ids)-prefix<=87:rejected['suffix_over_87']+=1;continue
            gold=row['expected'][key];assert gold in criteria
            record=p.prepare(dict(id=f'dis_{len(records):03d}',state=row['state'],question=q['instructions'],options=options,gold=gold))
            assert record['token_ids']==ids and record['prefix_tokens']==prefix
            cache=R/('artifacts/query-packing-v3/prefix-v2/queries/cache.json' if prefix==27 else 'artifacts/decision-index-v1/prefix/queries/cache.json')
            cached=json.loads(cache.read_text())['token_ids'];assert ids[:prefix]==cached
            record.update(dataset=family,source_id=row['id']);records.append(record);selected.append(row);count+=1
            if count==20:break
        assert count==20,(family,count,rejected)
        inventory[family]=dict(source_rows=len(rows),scanned_in_hash_order=scanned,selected=20,excluded_in_scanned=dict(rejected))
    source=D/'selected-source-rows.jsonl';source.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in selected))
    fixture=dict(scope='Short-input 100-question partial evaluation; maximum 87 suffix tokens; not the full Decision Index',
                 seed=seed,selection='SHA256(seed:family:group_id), first 20 eligible per dataset; eligibility only by input token length, never predictions or gold',
                 model_lock_sha256=e.sha(R/'MODEL_LOCK.json'),official_kit_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=KIT,text=True).strip(),
                 bias='Conditioning on short prompts favors concise/easier examples; no long-context, retrieval or tool-catalog claim.',
                 max_suffix_tokens=87,records=records,inventory=inventory,
                 source_hashes={str(path.relative_to(R)):e.sha(path) for path in source_paths+[source,R/'scripts/prepare_text.py',pathlib.Path(__file__),KIT/'hub/excluded-questions.json']})
    e.atomic_json(D/'inputs.json',fixture)
    print(json.dumps(dict(total=len(records),inventory=inventory,min_tokens=min(len(r['token_ids']) for r in records),max_tokens=max(len(r['token_ids']) for r in records))))

if __name__=='__main__':main()
