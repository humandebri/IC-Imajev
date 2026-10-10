import unittest

from prepare_text import TextPreparer


class TextTokenLimitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preparer = TextPreparer()

    def test_real_text_at_paid_sequence_boundaries(self):
        case = dict(id='token-limit', state='Evidence more', question='Approve?', options=['yes', 'no'])
        seed = self.preparer.prepare(case)
        for total in [512, 513, 1024, 1025]:
            state = 'Evidence' + ' more' * (total - len(seed['token_ids']) + 1)
            prompt = seed['prompt'].replace('State: "Evidence more"', f'State: "{state}"', 1)
            ids = self.preparer.tokenizer.encode(self.preparer.render(prompt), add_special_tokens=False)
            self.assertEqual(len(ids), total, 'boundary fixture must be actual tokenized text')
            with self.subTest(total=total):
                if total <= 1024:
                    result = self.preparer.prepare(dict(case, state=state))
                    self.assertEqual(result['token_ids'], ids)
                    self.assertEqual(result['prefix_tokens'], 5)
                else:
                    with self.assertRaisesRegex(ValueError, '1..1024 tokens; got 1025'):
                        self.preparer.prepare(dict(case, state=state))


if __name__ == '__main__':
    unittest.main()
