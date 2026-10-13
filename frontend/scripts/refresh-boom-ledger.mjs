// Read-only ledger snapshot for the #653 counterfactual demo.
import { Actor, HttpAgent } from '@icp-sdk/core/agent';
import { IDL } from '@icp-sdk/core/candid';
import { Principal } from '@icp-sdk/core/principal';
import { writeFile } from 'node:fs/promises';
const root = 'xjngq-yaaaa-aaaaq-aabha-cai';
const ledger = 'vtrom-gqaaa-aaaaq-aabia-cai';
const recipient = '4mpab-ayg5c-m2frb-vjja5-b2qbe-5rxgv-33ova-ddjgm-egfav-br5hu-gqe';
const agent = await HttpAgent.create({ host: 'https://icp-api.io' });
const factory = () => IDL.Service({
  icrc1_total_supply: IDL.Func([], [IDL.Nat], ['query']),
  icrc1_balance_of: IDL.Func([IDL.Record({ owner: IDL.Principal, subaccount: IDL.Opt(IDL.Vec(IDL.Nat8)) })], [IDL.Nat], ['query']),
  icrc1_decimals: IDL.Func([], [IDL.Nat8], ['query']),
});
const actor = Actor.createActor(factory, { agent, canisterId: ledger });
const startedAt = new Date().toISOString();
const [supply, balance, decimals] = await Promise.all([
  actor.icrc1_total_supply(),
  actor.icrc1_balance_of({ owner: Principal.fromText(recipient), subaccount: [] }),
  actor.icrc1_decimals(),
]);
if (decimals !== 8 || balance > supply) throw new Error('Unexpected ledger decimals or balance');
const snapshot = { rootCanisterId: root, ledgerCanisterId: ledger, recipient, subaccount: null,
  startedAt, completedAt: new Date().toISOString(), decimals,
  totalSupplyE8s: supply.toString(), recipientBalanceE8s: balance.toString(),
  methods: ['icrc1_total_supply', 'icrc1_balance_of', 'icrc1_decimals'],
  scenario: 'Additional mint now, not a reconstruction of proposal-time holdings. Separate queries, not an atomic snapshot.' };
await writeFile(new URL('../data/boom-653-ledger.json', import.meta.url), JSON.stringify(snapshot, null, 2) + '\n');
console.log(JSON.stringify(snapshot, null, 2));
