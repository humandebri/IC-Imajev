#!/usr/bin/env python3
"""Apply the measured POC tariff to an explicitly selected local paid canister."""
import argparse,json
from pathlib import Path
from paid_update_transport import PaidTransport
R=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--canister',required=True)
 ap.add_argument('--directory',required=True,help='New output directory for payment configuration evidence')
 args=ap.parse_args();dest=R/args.directory;dest.mkdir(parents=True,exist_ok=False)
 wire=PaidTransport(dest,args.canister);old=wire.call('paid_config')['result']
 config=json.loads((R/'examples/paid-inference-caller/poc-config.json').read_text());config['version']=old['version']+1
 reply=wire.call('configure_paid',config)['result']
 if 'Ok' not in reply:raise RuntimeError(reply)
 actual=wire.call('paid_config')['result'];assert actual==config
 (dest/'applied.json').write_text(json.dumps(dict(canister=args.canister,network='local',before=old,after=actual),indent=2)+'\n')
 print(json.dumps(actual,indent=2))
if __name__=='__main__':main()
