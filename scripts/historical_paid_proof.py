"""Prevent frozen prefix27 experiments from executing against the current API."""


def historical_only():
    raise SystemExit(
        'Historical prefix27 proof; execution is retired. Saved results remain under artifacts/. '
        'Use scripts/prove_paid_prefix5_local.py and scripts/measure_paid_prefix5.py '
        'with a current five-token build. See docs/PAID_PREFIX5_VERIFICATION.md.'
    )
