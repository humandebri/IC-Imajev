#!/usr/bin/env python3
"""Historical frozen-artifact optimized builder; not the public source release."""
if __name__ == '__main__':
    from script_usage import require_historical_cli
    require_historical_cli('reproduce_frozen_optimized.py', 'Historical frozen-artifact optimized build')
    from build_latest_common_prefix27 import main
    main()
