"""Frozen short-v2 preparation for historical prefix26/27 measurements.

Keep these prompts separate from the current prefix5 paid/browser contract.
"""
from prepare_text import TextPreparer as CurrentTextPreparer, evidence_header, SHORT_INSTRUCTIONS

PROMPT_LAYOUT = 'text-only-short-v2'


def shorten_header(header):
    return SHORT_INSTRUCTIONS + '\n' + evidence_header(header)


class TextPreparer(CurrentTextPreparer):
    prompt_layout = PROMPT_LAYOUT
    max_tokens = 512

    def prepare_state(self, case):
        return case.get('state', {})

    def prepare_header(self, header):
        return shorten_header(header)

    def count_prefix(self, rendered, ids, state):
        prefix = rendered.partition('\nState: ')[0] + '\nState: '
        if isinstance(state, str):
            prefix += '"'
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        count = 0
        for expected, actual in zip(prefix_ids, ids):
            if expected != actual:
                break
            count += 1
        return count
