//! Synthetic A/B harness. Both revisions use this identical fixture and wrapper.
use imajev_runtime::{int8_kernel as block, int8_token_kernel as token};
use std::cell::RefCell;
struct Fixture { mode: usize, n: usize, cols: usize, x: Vec<f32>, w: Vec<i8>, scales: Vec<f32>, out: Vec<f32> }
thread_local! { static FIXTURE: RefCell<Option<Fixture>> = const { RefCell::new(None) }; }
#[no_mangle]
pub extern "C" fn setup(mode: usize, n: usize, cols: usize, pattern: usize) {
    assert!(mode <= 6 && (1..=132).contains(&n) && matches!(cols, 256 | 512 | 2560 | 9216));
    let mut seed = 0x537a9201u32;
    let mut next = || { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); seed };
    let x = (0..n*cols).map(|i| match pattern {
        0 => ((next() >> 16) % 255) as f32 / 41. - 3.,
        1 => if i%2==0 {127.} else {-127.},
        2 => 0.,
        3 => [f32::from_bits(1),-f32::from_bits(1),0.][i%3],
        4 => 127.,
        5 => -127.,
        _ => panic!("pattern"),
    }).collect();
    let w = (0..32*cols).map(|_| if pattern>=4 {-128} else {((next() >> 16) % 255) as i16 - 127}).map(|v|v as i8).collect();
    let scales = (0..32).map(|i| (i%7+1) as f32 * 0.0017).collect();
    FIXTURE.with(|f| *f.borrow_mut() = Some(Fixture {mode,n,cols,x,w,scales,out:Vec::new()}));
}
#[no_mangle]
pub extern "C" fn run() {
    FIXTURE.with(|f| {
        let mut f=f.borrow_mut(); let f=f.as_mut().unwrap();
        f.out=if f.mode==6 {
            token::project(&token::quantize(&f.x,f.n,f.cols).unwrap(),&f.w,&f.scales,32).unwrap()
        } else {
            let q=block::quantize_rows(&f.x,f.n,f.cols).unwrap();
            match f.mode {
                0=>block::project(&q,&f.w,&f.scales,32),
                1=>block::project_column16(&q,&f.w,&f.scales,32),
                2=>block::project_column16_token48(&q,&f.w,&f.scales,32),
                3=>block::project_dot_scale(&q,&f.w,&f.scales,32),
                4=>block::project_balanced44(&q,&f.w,&f.scales,32),
                5=>block::project_column32_balanced(&q,&f.w,&f.scales,32),
                _=>unreachable!(),
            }.unwrap()
        };
    });
}
#[no_mangle]
pub extern "C" fn output_ptr()->usize {FIXTURE.with(|f|f.borrow().as_ref().unwrap().out.as_ptr() as usize)}
#[no_mangle]
pub extern "C" fn output_len()->usize {FIXTURE.with(|f|f.borrow().as_ref().unwrap().out.len())}
#[cfg(target_arch="wasm32")]
mod ic {
    #[link(wasm_import_module="ic0")]
    extern "C" {
        fn performance_counter(kind:i32)->i64;
        fn msg_arg_data_size()->i32;
        fn msg_arg_data_copy(dst:i32,offset:i32,size:i32);
        fn msg_reply_data_append(src:i32,size:i32);
        fn msg_reply();
    }
    unsafe fn reply(bytes:&[u8]) {msg_reply_data_append(bytes.as_ptr() as i32,bytes.len() as i32);msg_reply();}
    #[export_name="canister_update configure"]
    pub extern "C" fn configure() {unsafe {
        assert_eq!(msg_arg_data_size(),16);let mut b=[0u8;16];msg_arg_data_copy(b.as_mut_ptr() as i32,0,16);
        let arg=|i|u32::from_le_bytes(b[i..i+4].try_into().unwrap()) as usize;
        super::setup(arg(0),arg(4),arg(8),arg(12));reply(&[]);
    }}
    #[export_name="canister_query measure"]
    pub extern "C" fn measure() {unsafe {
        let before=performance_counter(0);super::run();let instructions=(performance_counter(0)-before) as u64;
        let mut hash=14695981039346656037u64;
        super::FIXTURE.with(|f|for v in &f.borrow().as_ref().unwrap().out {for byte in v.to_le_bytes(){hash=(hash^byte as u64).wrapping_mul(1099511628211);}});
        let mut b=Vec::new();b.extend(instructions.to_le_bytes());b.extend(hash.to_le_bytes());
        b.extend((super::output_len() as u32).to_le_bytes());b.extend((core::arch::wasm32::memory_size(0) as u32).to_le_bytes());reply(&b);
    }}
}
