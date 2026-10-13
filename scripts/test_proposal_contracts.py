"""Evidence boundaries for the replacement snapshot and exact-encoding utilities."""
import json
from fractions import Fraction
import unittest
from tools.proposal_assessment.core import assess, old_parameter, run_advisory
from tools.proposal_assessment.vote import make_vote_task, predict_vote
from tools.proposal_assessment.compact_units import encode
from tools.proposal_assessment.improve_binary import terminating_decimal, approval_evidence_gate
from tools.proposal_assessment.binary_benchmark import binary_decision
from tools.proposal_assessment.budget_policy import evidence_gate

FIELD = 'neuron_minimum_dissolve_delay_to_vote_seconds'

def proposal(old='Some(86400)', new=172800):
    return dict(id=1,proposal_action_type='ManageNervousSystemParameters',proposal_action_payload={FIELD:new},
                payload_text_rendering=f'## Current nervous system parameters:\n{FIELD}: {old}',
                proposal_title='Example',summary='Change the minimum lock.')

def task(rows):
    return dict(state=json.dumps(dict(action='ManageNervousSystemParameters',changes_or_requests=rows)))

class Contracts(unittest.TestCase):
    def test_zero_transition_does_not_invent_a_ratio(self):
        row = assess(proposal('Some(0)'))['items'][0]
        self.assertEqual(row['direction'],'increase')
        self.assertIsNone(row['magnitude_ratio'])
        self.assertEqual(row['review'],'review')

    def test_duplicate_old_field_is_unavailable(self):
        p = proposal()
        p['payload_text_rendering'] += f'\n{FIELD}: Some(1)'
        self.assertEqual(old_parameter(p,FIELD),(None,None))
        self.assertIn('previous_value',assess(p)['items'][0]['unknowns'])

    def test_rendering_is_not_verified_chain_state(self):
        report = assess(proposal())
        self.assertFalse(report['execution_authorization'])
        self.assertFalse(report['items'][0]['evidence'][1]['chain_state_verified'])
        self.assertIn('not chain-verified',make_vote_task(report)['state'])

    def test_boolean_and_uint64_overflow_are_invalid_numeric_values(self):
        for value in (True,-1,2**64):
            with self.subTest(value=value):
                row = assess(proposal(new=value))['items'][0]
                self.assertEqual(row['status'],'insufficient_data')
                self.assertIn('valid_proposed_uint64',row['unknowns'])

    def test_unit_conversion_is_exact_and_keeps_all_fields(self):
        rows = [dict(field='max_dissolve_delay_seconds',previous=2629800,proposed=2629800000),
                dict(field='neuron_minimum_stake_e8s',previous=100001,proposed=100000000),
                dict(field='wait_for_quiet_deadline_increase_seconds',previous=86400,proposed=86401)]
        text,facts = encode(task(rows))
        self.assertIn('30.4375→30437.5',text)
        self.assertIn('0.00100001→1',text)
        self.assertIn('1day→86401second',text)
        self.assertEqual({(r['field'],r['previous'],r['proposed']) for r in facts},
                         {(r['field'],r['previous'],r['proposed']) for r in rows})
        self.assertIsNone(terminating_decimal(Fraction(1,3)))

    def test_missing_duplicate_and_unsupported_changes_cannot_be_dropped(self):
        row = dict(field=FIELD,previous=1,proposed=2)
        for rows in ([],[row,row],[dict(row,previous=None)],[dict(row,field='unsupported')]):
            with self.subTest(rows=rows),self.assertRaises(ValueError):
                encode(task(rows))

    def test_approval_requires_improvement_without_conflicting_barrier(self):
        lower = dict(field=FIELD,previous=172800,proposed=86400)
        higher = dict(field='neuron_minimum_stake_e8s',previous=1,proposed=2)
        self.assertEqual(approval_evidence_gate(task([lower]),'approve'),('approve',None))
        self.assertEqual(approval_evidence_gate(task([lower,higher]),'approve')[0],'hold')
        self.assertEqual(approval_evidence_gate(task([lower,higher]),'reject'),('reject',None))

    def test_generic_payload_conflict_requires_hold(self):
        t = task([dict(field='generic_call',unknowns=[],payload_sha256='a',rendered_payload_sha256='b')])
        result = evidence_gate(t)
        self.assertEqual(result['label'],'hold')
        self.assertTrue(result['payload_conflict'])
        self.assertFalse(result['execution_authorization'])

    def test_nonfinite_model_output_and_invalid_evidence_stay_unavailable(self):
        class Adapter:
            def metadata(self):
                return dict(name='fake')
            def predict(self,task):
                return dict(label=task['options'][0],logits=[float('nan')]*len(task['options']))
        self.assertEqual(predict_vote(Adapter(),make_vote_task(assess(proposal())))['status'],'unavailable')
        class BadSpan(Adapter):
            def predict(self,task):
                return dict(label='present',evidence=[dict(source='/summary',start=0,end=10000)])
        report = run_advisory(assess(proposal()),BadSpan())
        self.assertTrue(all(row['status']=='unavailable' for row in report['advisory']))
        self.assertFalse(report['execution_authorization'])

    def test_ties_extreme_scores_and_threshold_validation(self):
        self.assertEqual(binary_decision([1,1],.5),('hold',.5))
        self.assertEqual(binary_decision([1e308,-1e308],.6),('approve',1.))
        for logits,threshold in (([True,0],.6),([float('nan'),0],.6),([0,1],True),([0,1],.49)):
            with self.assertRaises(ValueError):
                binary_decision(logits,threshold)

if __name__=='__main__':
    unittest.main()
