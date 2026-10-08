#!/usr/bin/env python3
"""Exact-value and rejection tests for built-in example compaction."""
import json
from pathlib import Path
import unittest
from prepare_browser_examples import compact_mint, compact_parameters, parameter_rows

ROOT = Path(__file__).resolve().parents[1]
SOURCE = {sample['id']: sample['input']['state'] for sample in json.loads((ROOT / 'frontend/data/boom-examples.json').read_text())}


class ExampleCompactionTests(unittest.TestCase):
    def test_parameter_fields_and_values_preserved(self):
        for name in ('boom-620', 'boom-617'):
            rows = parameter_rows(SOURCE[name])
            _, audit = compact_parameters(SOURCE[name])
            expected = {row['field']: (row['previous'], row['proposed']) for row in rows}
            actual = {fact['field']: (fact['previous'], fact['proposed']) for fact in audit['facts']}
            self.assertEqual(actual, expected)
        self.assertEqual(compact_parameters(SOURCE['boom-617'])[1]['facts'][0]['rendered_values'], ['30.4375', '30437.5'])

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

    def test_inconsistent_mint_and_extra_evidence_rejected(self):
        for text in (SOURCE['boom-653'].replace('25000000000000000', '25000000000000001'),
                     SOURCE['boom-653'] + '\nTotal supply: 100',
                     SOURCE['boom-653'].replace('## Memo: 0', '## Memo: -1')):
            with self.assertRaises(ValueError):
                compact_mint(text)


if __name__ == '__main__':
    unittest.main()
