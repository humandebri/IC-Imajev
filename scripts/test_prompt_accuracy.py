"""Check paired evaluation denominators, abstention errors, and option mapping."""
import pathlib,unittest
from unittest.mock import patch
import evaluate_prompt_accuracy as e
class Tests(unittest.TestCase):
 def row(self,id='x',offset=0,variant='original',gold='yes',pred='yes'):
  return dict(id=id,offset=offset,variant=variant,category='control',gold=gold,prediction=pred,correct=gold==pred,gold_probability=.7)
 def test_unknown_and_false_abstention_are_distinct_errors(self):
  rows=[self.row(),self.row(pred='__unknown__'),self.row(gold='__unknown__',pred='__unknown__'),self.row(gold='__unknown__',pred='yes')]
  m=e.metrics(rows)
  self.assertEqual((m['total'],m['correct'],m['known_total'],m['known_correct'],m['unknown_total'],m['unknown_correct']),(4,2,2,1,2,1))
  self.assertEqual((m['false_abstentions'],m['unsupported_answers'],m['answered_total'],m['answered_correct']),(1,1,2,1))
 def test_rotations_do_not_inflate_primary_accuracy(self):
  pairs=[dict(id='x',offset=o,category='control',gold='yes') for o in (0,1)]
  rows=[self.row(offset=0,pred='no'),self.row(offset=1),self.row(offset=0,variant='short'),self.row(offset=1,variant='short')]
  s=e.summarize(pairs,rows)
  self.assertEqual(s['primary_distinct_cases']['original']['total'],1)
  self.assertEqual(s['all_ordered_cases']['original']['total'],2)
  self.assertEqual(len(s['order_changes']),1)
  self.assertEqual(len(s['improvements']),1)
  self.assertEqual(len(s['regressions']),0)
 def test_incomplete_pair_is_not_counted_as_answer_change(self):
  pairs=[dict(id='x',offset=0,category='control',gold='yes')]
  self.assertEqual(e.summarize(pairs,[self.row()])['paired_comparisons'],[])
 def test_gold_probability_follows_option_value_after_reordering(self):
  r=dict(wasm_sha256=e.MODULE,input_hash='hash',comparison=dict(typed_output_valid=True,gold='yes'),tokens=92,processed_tokens=65,query_count=39,replayed_queries=0,decision_query={'ok':{'decision':dict(abstained=False,value='yes',probabilities=[.2,.6],unknown_probability=.2)}})
  record=dict(input_sha256='hash',options=['no','yes']);pair=dict(id='x',offset=1,category='control',gold='yes')
  with patch.object(e,'sha',return_value='digest'):
   row=e.extract(r,record,pair,'short',e.ROOT/'nonexistent-evaluation-test')
  self.assertEqual(row['gold_probability'],.6);self.assertTrue(row['correct'])
if __name__=='__main__':unittest.main()
