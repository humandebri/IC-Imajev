const fs=require('fs'),path=require('path'),crypto=require('crypto');
const root=path.resolve(__dirname,'..'),d=path.join(root,'artifacts/worker-wrapper-checkpoint-tool-v1');
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
function check(v,m){if(!v)throw Error(m);}
let replies=0,reads=0;
const instance=new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(path.join(d,'test-patched.wasm'))),{ic0:{
 msg_reply:()=>{replies++;},
 performance_counter:(which)=>{check(which===0,'counter zero');check(replies>reads,'checkpoint must follow reply');reads++;return BigInt(9000+reads);}
}});
const e=instance.exports;
e['canister_update inference_step']();
check(replies===1&&reads===1&&e.__imajev_worker_checkpoint()===9001n,'armed checkpoint');
e.set_mode(0);e['canister_update inference_step']();
check(replies===2&&reads===1&&e.__imajev_worker_checkpoint()===9001n,'unarmed request preserves previous checkpoint');
e.set_mode(1);e['canister_update inference_step']();
check(replies===3&&reads===2&&e.__imajev_worker_checkpoint()===9002n,'subsequent armed worker records new checkpoint');
const full=path.join(root,'artifacts/paid-wrapper-checkpoint-v1/build'),report=JSON.parse(fs.readFileSync(path.join(full,'report.json')));
check(sha(path.join(full,'full.wasm'))===report.module,'full module identity');
check(WebAssembly.validate(fs.readFileSync(path.join(full,'full.wasm'))),'full engine validation');
for(const [p,h]of Object.entries(report.source_hashes))check(sha(path.join(root,p))===h,p);
const files=[__filename,path.join(d,'test.wat'),path.join(d,'test.wasm'),path.join(d,'test-patched.wasm'),path.join(d,'test-patch.json'),path.join(full,'report.json')];
fs.writeFileSync(path.join(d,'execution-report.json'),JSON.stringify({complete:true,after_reply_checkpoint_verified:true,unarmed_calls_preserve_counter:true,subsequent_armed_worker_replaces_counter:true,full_candidate_node_validation:true,ic_metering_verified:false,whole_message_upper_bound_proven:false,source_hashes:Object.fromEntries(files.map(p=>[path.relative(root,p),sha(p)]))},null,2)+'\n');
console.log('PASS after-reply recording, unarmed preservation, next-worker recording, full Wasm validation');
