#!/usr/bin/env python3
"""Compare replacement code against a separately archived pre-change package."""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from proposal_snapshots import read_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    package = args.baseline/'tools/proposal_assessment'
    spec = importlib.util.spec_from_file_location('baseline_assessment',package/'__init__.py',submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    names = ('core','vote','routing','budget_policy','improve_binary','compact_units')
    before = {name:importlib.import_module('baseline_assessment.'+name) for name in names}
    after = {name:importlib.import_module('tools.proposal_assessment.'+name) for name in names}
    manifest = json.loads((args.archive/'manifest.json').read_text())
    cases = []
    for record in manifest['records']:
        _,proposal = read_snapshot(args.archive,record,manifest['sns_root'])
        facts = [package['core'].assess(proposal) for package in (before,after)]
        assert facts[0]==facts[1], ('facts',record['proposal_id'],facts)
        tasks = [package['vote'].make_vote_task(fact) for package,fact in zip((before,after),facts)]
        assert tasks[0]==tasks[1], ('vote input',record['proposal_id'],tasks)
        for name,method in (('routing','select_task'),('budget_policy','evidence_gate')):
            assert getattr(before[name],method)(tasks[0])==getattr(after[name],method)(tasks[1]),(name,record['proposal_id'])
        assert before['core'].make_tasks(facts[0])==after['core'].make_tasks(facts[1])
        encodings = 0
        for name,method,options in (('improve_binary','encode_state',{}),('improve_binary','compact_exact_state',{}),
                                    ('compact_units','encode',{}),('compact_units','encode',{'ratio_words':True})):
            outputs = []
            for package,task in zip((before,after),tasks):
                try:
                    outputs.append(('ok',getattr(package[name],method)(task,**options)))
                except (ValueError,KeyError) as error:
                    outputs.append(('error',type(error).__name__))
            assert outputs[0]==outputs[1], (name,method,record['proposal_id'],outputs)
            encodings += outputs[0][0]=='ok'
        cases.append(dict(proposal_id=record['proposal_id'],facts_equal=True,task_bytes_equal=True,successful_encoding_checks=encodings))
    args.output.write_text(json.dumps(dict(complete=True,scope='Frozen snapshot facts/tasks/encodings; no new inference',cases=cases),indent=2)+'\n')
    print(json.dumps(dict(complete=True,proposals=len(cases))))

if __name__=='__main__':
    main()
