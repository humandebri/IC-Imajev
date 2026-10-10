#!/usr/bin/env python3
"""Exact-value and rejection tests for built-in example compaction."""
import json
from pathlib import Path
import unittest
from prepare_browser_examples import compact_mint, compact_parameters, parameter_rows, compact_treasury_transfer, attach_ledger, attach_neurons, analyze_neurons

ROOT = Path(__file__).resolve().parents[1]
SOURCE = {sample['id']: sample['input']['state'] for sample in json.loads((ROOT / 'frontend/data/boom-examples.json').read_text())}
SOURCE['boom-620'] = 'SNS parameters adjustment\n\n# Proposal to change nervous system parameters:\n## Current nervous system parameters:\n    neuron_minimum_dissolve_delay_to_vote_seconds: Some(86400,),\n## New nervous system parameters:\n    neuron_minimum_dissolve_delay_to_vote_seconds: Some(172800,),'



class ExampleCompactionTests(unittest.TestCase):
    def test_parameter_fields_and_values_preserved(self):
        for name in ('boom-620', 'boom-617'):
            rows = parameter_rows(SOURCE[name])
            _, audit = compact_parameters(SOURCE[name])
            expected = {row['field']: (row['previous'], row['proposed']) for row in rows}
            actual = {fact['field']: (fact['previous'], fact['proposed']) for fact in audit['facts']}
            self.assertEqual(actual, expected)
        maximum = compact_parameters(SOURCE['boom-617'])[1]['facts'][0]
        self.assertEqual(maximum['unit_group'], 'hours')
        self.assertEqual(maximum['rendered_values'], ['730.5', '730500'])
        self.assertEqual(compact_parameters(SOURCE['boom-620'])[0],
                         'The minimum voting lock duration changes from 1 day to 2 days.')

    def test_no_unknown_or_duplicate_parameter_text_dropped(self):
        for text in (SOURCE['boom-620'] + '\nUnparsed evidence',
                     SOURCE['boom-620'].replace('## New nervous system parameters:',
                       'neuron_minimum_dissolve_delay_to_vote_seconds: Some(86400,),\n## New nervous system parameters:'),
                     SOURCE['boom-620'].replace('neuron_minimum_dissolve_delay_to_vote_seconds', 'unsupported_field')):
            with self.assertRaises(ValueError):
                compact_parameters(text)

    def test_mint_aliases_and_exact_amount(self):
        state, audit = compact_mint(SOURCE['boom-653'])
        self.assertIn('250M SNS tokens', state)
        self.assertEqual(audit['amount_e8s'], '25000000000000000')
        self.assertEqual(len(audit['identifiers']), 1)
        self.assertEqual(audit['principal'], audit['account'])
        self.assertEqual(audit['identifiers'][0]['value'], audit['principal'])
        self.assertEqual(audit['memo'], 0)
        _, changed = compact_mint(SOURCE['boom-653'].replace('## Target account: ' + audit['account'], '## Target account: another-account'))
        self.assertEqual(len(changed['identifiers']), 2)
        self.assertNotEqual(changed['identifiers'][0]['alias'], changed['identifiers'][1]['alias'])

    def test_ledger_bounds_enclose_exact_readings(self):
        state, audit = attach_ledger(*compact_mint(SOURCE['boom-653']))
        snapshot, bounds = audit['ledger_snapshot'], audit['input_bounds']
        self.assertNotIn('unknown', state)
        for field, key, scale in [('supply', 'totalSupplyE8s', bounds['supply_scale_e8s']),
                                  ('balance', 'recipientBalanceE8s', bounds['balance_scale_e8s'])]:
            lower, upper = bounds[field]
            self.assertLessEqual(lower * scale, int(snapshot[key]))
            self.assertGreaterEqual(upper * scale, int(snapshot[key]))
        # The ranges suffice for this comparison even at their worst endpoints.
        mint = int(audit['amount_e8s'])
        derived = audit['derived']
        post_balance = int(snapshot['recipientBalanceE8s']) + mint
        post_supply = int(snapshot['totalSupplyE8s']) + mint
        self.assertEqual(int(derived['post_supply_e8s']), post_supply)
        self.assertEqual(int(derived['post_balance_e8s']), post_balance)
        lo, hi = derived['share_percent_bounds']
        self.assertLessEqual(lo * post_supply, 100 * post_balance)
        self.assertGreaterEqual(hi * post_supply, 100 * post_balance)
        self.assertLess(2 * (bounds['balance'][1] * 10**8 + mint), bounds['supply'][0] * 10**14 + mint)
        _, audit = compact_mint(SOURCE['boom-653'])
        audit['principal'] = 'another-account'
        with self.assertRaises(ValueError):
            attach_ledger('', audit)

    def test_title_and_treasury_action_preserved(self):
        state, audit = compact_treasury_transfer(SOURCE['boom-584'])
        self.assertIn('SNS Metadata Adjustment', state)
        self.assertIn('send 20M BOOM from DAO treasury', state)
        self.assertEqual(audit['status'], 'REJECTED')
        self.assertEqual(audit['amount_e8s'], '2000000000000000')
        self.assertEqual(audit['from_treasury'], 2)
        self.assertEqual(audit['identifiers'][0]['value'],
                         'jrnhz-6ekxv-2fffs-wfcgt-l3pe7-456id-heznf-xyf64-nykjq-4jyso-zae')
        for changed in (SOURCE['boom-584'].replace('2000000000000000', '2000000000000001'),
                        SOURCE['boom-584'] + 'extra evidence'):
            with self.assertRaises(ValueError):
                compact_treasury_transfer(changed)

    def test_neuron_thresholds_and_dissolving_time(self):
        snapshot = dict(rootCanisterId='xjngq-yaaaa-aaaaq-aabha-cai', referenceTimestampSeconds=100,
                        totalNeurons=4, neurons=[
                            dict(id='a', voting_power='100', dissolve_state={'DissolveDelaySeconds': 86400}),
                            dict(id='b', voting_power='200', dissolve_state={'DissolveDelaySeconds': 1728000000}),
                            dict(id='c', voting_power='300', dissolve_state={'WhenDissolvedTimestampSeconds': 86500}),
                            dict(id='d', voting_power='0', dissolve_state={'DissolveDelaySeconds': 1728000000})])
        analysis = analyze_neurons(snapshot, 86400, 1728000000)
        self.assertEqual(analysis['before']['count'], 3)
        self.assertEqual(analysis['after']['count'], 1)
        self.assertEqual(analysis['excluded_neurons'], 2)
        self.assertEqual(analysis['after']['largest_share_bps_floor'], 10000)
        self.assertEqual(analyze_neurons(snapshot, 86400, 1728000001)['after']['largest_share_bps_floor'], None)
        snapshot['neurons'][1]['id'] = 'a'
        with self.assertRaises(ValueError):
            analyze_neurons(snapshot, 86400, 1728000000)

    def test_participation_source_and_share_calculation(self):
        state, audit = attach_neurons(*compact_parameters(SOURCE['boom-617']))
        retained = audit['participation_snapshot']
        self.assertIn('eligible neurons', state)
        self.assertIn('Current-neuron simulation, not historical', state)
        self.assertIn('No relocking.', state)
        self.assertNotIn('attack', state)
        for group in ('before', 'after'):
            values = retained['analysis'][group]
            total, largest = int(values['indexed_voting_power']), int(values['largest_neuron_voting_power'])
            if total:
                bps = values['largest_share_bps_floor']
                self.assertLessEqual(bps * total, largest * 10000)
                self.assertGreater((bps + 1) * total, largest * 10000)
        self.assertEqual(len(retained['sha256']), 64)

    def test_inconsistent_mint_and_extra_evidence_rejected(self):
        for text in (SOURCE['boom-653'].replace('25000000000000000', '25000000000000001'),
                     SOURCE['boom-653'] + '\nTotal supply: 100',
                     SOURCE['boom-653'].replace('## Memo: 0', '## Memo: -1')):
            with self.assertRaises(ValueError):
                compact_mint(text)


if __name__ == '__main__':
    unittest.main()
