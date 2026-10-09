"""Pinned-tokenizer regressions for the historical preparation contract."""
import unittest

from prepare_text import TextPreparer as CurrentTextPreparer, SHORT_INSTRUCTIONS
from prepare_text_legacy import TextPreparer, shorten_header
from vision_decision.contracts import ChoiceField, Option
from vision_decision.scoring import compile_question


class LegacyPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preparer = TextPreparer()
        cls.current = CurrentTextPreparer()

    def case(self, state):
        return dict(id='legacy-regression', state=state, question='Which option?', options=['yes', 'no'])

    def test_frozen_main_token_hashes_and_prefix_boundaries(self):
        # These hashes come from main before the prefix5 migration.
        fixtures = [
            ('Evidence', 27, 'f9b9a5d90ff22c4637dc8da647377307aa804241a44729fb37a8e419002a3e0b'),
            ({'text': 'Evidence'}, 26, '93b8866708b46b9130faccd91beb75ad9a458b0ded1148c393c1306916c82961'),
            ('', 26, '450c20d50ddde5823aa81521008d21ed6bb0c1545fdf5c213518c1873a239053'),
            ({}, 26, '7402f245e0fe4c2cddcc2fb7b7bdaf13e57b0c02766602182ed3459d51f55df6'),
        ]
        for state, prefix, digest in fixtures:
            with self.subTest(state=state):
                record = self.preparer.prepare(self.case(state))
                self.assertEqual(record['input_sha256'], digest)
                self.assertEqual(record['prefix_tokens'], prefix)
                self.assertEqual(record['prompt_layout'], 'text-only-short-v2')
        missing = self.case({})
        del missing['state']
        self.assertEqual(self.preparer.prepare(missing), self.preparer.prepare(self.case({})))

    def test_subset_header_matches_preparation_without_changing_evidence(self):
        state = SHORT_INSTRUCTIONS
        case = self.case(state)
        field = ChoiceField(id=case['id'].replace('-', '_'), type='choice', question=case['question'],
                            options=[Option(value=value) for value in case['options']])
        header, _, texts = compile_question(field, state, 'standard')
        texts[-1] = 'unknown'
        prompt = shorten_header(header) + '\n'.join(f'{code}: {text}' for code, text in zip('ABC', texts))
        self.assertEqual(self.preparer.prepare(case)['prompt'], prompt)
        self.assertEqual(prompt.count(SHORT_INSTRUCTIONS), 2)
        with self.assertRaisesRegex(ValueError, 'unverified standard prompt instructions'):
            shorten_header('unexpected\nState: evidence')

    def test_legacy_limit_stays_512_while_current_accepts_513(self):
        case = self.case('Evidence more')
        seed = self.preparer.prepare(case)
        for total in [512, 513]:
            state = 'Evidence' + ' more' * (total - len(seed['token_ids']) + 1)
            with self.subTest(total=total):
                if total == 512:
                    self.assertEqual(len(self.preparer.prepare(self.case(state))['token_ids']), total)
                else:
                    with self.assertRaisesRegex(ValueError, '1..512 tokens; got 513'):
                        self.preparer.prepare(self.case(state))
                    self.assertEqual(self.current.prepare(self.case(state))['prefix_tokens'], 5)

    def test_current_contract_still_rejects_object_states(self):
        with self.assertRaisesRegex(ValueError, 'state must be a string'):
            self.current.prepare(self.case({'text': 'Evidence'}))
        record = self.current.prepare(self.case('Evidence'))
        self.assertEqual(record['prefix_tokens'], 5)
        self.assertEqual(record['prompt_layout'], 'text-only-state-prefix5-v1')
        self.assertFalse(record['prompt'].startswith(SHORT_INSTRUCTIONS))


if __name__ == '__main__':
    unittest.main()
