"""Select an exact reusable voting stem; variable values/questions are online."""
from pathlib import Path
import json
MODULE='2cfe5d3a1d2da9481d9446dbd0a877617de8db140f8347058b02a1dbb2c63a05'
VOTING38=(248045,846,198,56555,279,2420,5721,321,4087,279,3296,1608,279,10661,12521,13,3301,1132,279,3074,2904,1970,13,198,1349,25,328,27756,15209,70173,7383,4203,494,220,16,1834,310,220)

def select(root,token_ids,module,cache,packets):
    if module!=MODULE or tuple(token_ids[:38])!=VOTING38:
        return Path(cache),None if packets is None else Path(packets),'common'
    bank=Path(root)/'artifacts/voting-template-prefix-v1'
    metadata=json.loads((bank/'template.json').read_text())
    if tuple(metadata['token_ids'])!=VOTING38 or metadata['tokens']!=38:
        raise ValueError('voting template identity')
    return bank/'prefix/queries',bank/'packets','voting-fixed-stem38'
