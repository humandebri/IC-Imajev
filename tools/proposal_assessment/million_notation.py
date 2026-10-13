"""Exact decimal formatting; no experiment driver or external checkpoint."""
from fractions import Fraction
import re

NUMBER = re.compile(r'[0-9]+(?:\.[0-9]+)?\Z')

def exact_value(text):
    if not isinstance(text, str):
        raise ValueError('decimal text required')
    raw = text.removesuffix('M')
    if not NUMBER.fullmatch(raw):
        raise ValueError('unsigned decimal required')
    return Fraction(raw) * (1000000 if text.endswith('M') else 1)

def format_million(text):
    if type(text) is int:
        text = str(text)
    if not isinstance(text, str) or not NUMBER.fullmatch(text):
        raise ValueError('unsigned decimal required')
    whole, _, fractional = text.partition('.')
    if int(whole) < 1000000:
        return text
    digits = whole.zfill(7)
    integral, decimal = digits[:-6], (digits[-6:] + fractional).rstrip('0')
    return str(int(integral)) + ('.' + decimal if decimal else '') + 'M'
