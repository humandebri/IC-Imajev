use super::*;
#[ic_cdk::update(name = "prepareFixedPrefixCache")]
fn prepare_fixed_prefix_state(log:StateBytes,packet:StateBytes)->Result<(u32,u64,u64),String>{
 owner();let start=ic_cdk::api::performance_counter(0);
 let(count,bytes)=imajev_runtime::prefix_state_cache::prepare(&log,&packet)?;
 Ok((count,bytes,ic_cdk::api::performance_counter(0)-start))
}
