import copy,json,pathlib,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from decision_validation import validate_decision
class Tests(unittest.TestCase):
 def setUp(self):
  self.options=['yes','no'];self.valid=dict(value='yes',abstained=False,probabilities=[.8,.1],unknown_probability=.1,raw_logits=[1.,0.,-1.],instructions=5,calibration_version='p3-r2-s000291-authored')
 def test_valid_and_abstention(self):
  validate_decision(self.valid,self.options)
  r=dict(self.valid,value=None,abstained=True);validate_decision(r,self.options)
  del r['value']
  with self.assertRaises(ValueError):validate_decision(r,self.options)
 def test_invalid_output_rejected(self):
  cases={'probabilities':[[-.1,1.],[float('nan'),.1],[True,.1],[.8],[.8,.8]],'raw_logits':[[10**1000,0.,0.],[],[1.,float('inf'),0.],[1.,0.]],'unknown_probability':[-.1,1.1,True,float('nan')],'abstained':[1,True],'value':['other',1],'calibration_version':['other',None],'instructions':[-1,True,.1]}
  for key,values in cases.items():
   for value in values:
    with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_decision(dict(self.valid,**{key:value}),self.options)
  for key in self.valid:
   r=self.valid.copy();del r[key]
   with self.assertRaises(ValueError):validate_decision(r,self.options)
 def test_saved_real_decisions(self):
  for label in ['617','insufficient','maximum']:
   p=ROOT/f'artifacts/tail50/tail-proof-v1/{label}/report.json'
   if not p.exists():continue
   r=json.loads(p.read_text());validate_decision(r['decision_query']['ok']['decision'],r['queries'][-1]['decision_options'])
if __name__=='__main__':unittest.main()
