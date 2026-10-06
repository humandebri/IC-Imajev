//! Local-only prepared projection diagnostic; no production API contract.
use imajev_runtime::int8_token_kernel::{project, TokenRows};
use std::cell::RefCell;
mod laya {
    include!(concat!(env!("LAYA_KERNEL_FIXTURES"), "/laya_extracted.rs"));
}
const QKV: &[u8] = include_bytes!(concat!(env!("LAYA_KERNEL_FIXTURES"), "/qkv.bin"));
const WO: &[u8] = include_bytes!(concat!(env!("LAYA_KERNEL_FIXTURES"), "/wo.bin"));
struct State {
    n: usize,
    cols: usize,
    input: Vec<i8>,
    token: TokenRows,
    original_weights: Vec<i8>,
    padded_weights: Vec<i8>,
    scales: Vec<f32>,
    output: Vec<f32>,
}
thread_local! {static STATE: RefCell<Option<State>> = const { RefCell::new(None) };}
#[no_mangle]
pub extern "C" fn setup(n: usize, rows: usize, cols: usize) {
    assert!((1..=128).contains(&n) && rows == 512 && matches!(cols, 1024 | 2624));
    let bytes = if cols == 1024 { QKV } else { WO };
    assert_eq!(bytes.len(), rows * cols + rows * 4);
    let weights: Vec<i8> = bytes[..rows * cols].iter().map(|v| *v as i8).collect();
    let scales: Vec<f32> = bytes[rows * cols..]
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert!(scales.iter().all(|v| v.is_finite() && *v > 0.));
    let padded_cols = cols.div_ceil(256) * 256;
    let padded_tokens = n.div_ceil(8) * 8;
    let input: Vec<i8> = (0..n * cols)
        .map(|i| ((i * 19 % 255) as i32 - 127) as i8)
        .collect();
    let sx: Vec<f32> = (0..n).map(|t| (t + 1) as f32 / 257.).collect();
    let mut values = vec![0i16; padded_tokens * padded_cols];
    for t in 0..n {
        for c in 0..cols {
            values[t * padded_cols + c] = input[t * cols + c] as i16;
        }
    }
    let mut padded_weights = vec![0i8; rows * padded_cols];
    for r in 0..rows {
        padded_weights[r * padded_cols..r * padded_cols + cols]
            .copy_from_slice(&weights[r * cols..(r + 1) * cols]);
    }
    let mut token_scales = vec![1.; padded_tokens];
    token_scales[..n].copy_from_slice(&sx);
    STATE.with(|state| {
        *state.borrow_mut() = Some(State {
            n,
            cols,
            input,
            token: TokenRows {
                values,
                scales: token_scales,
                rows: n,
                cols: padded_cols,
            },
            original_weights: weights,
            padded_weights,
            scales,
            output: Vec::new(),
        })
    });
}
#[no_mangle]
pub extern "C" fn run(backend: usize) {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let s = state.as_mut().unwrap();
        s.output = match backend {
            0 => laya::project(
                &s.input,
                &s.token.scales[..s.n],
                &s.original_weights,
                &s.scales,
                s.n,
                512,
                s.cols,
            ),
            1 => project(&s.token, &s.padded_weights, &s.scales, 512).unwrap(),
            _ => panic!("backend"),
        };
        assert!(s.output.iter().all(|v| v.is_finite()));
    });
}
#[no_mangle]
pub extern "C" fn output_ptr() -> usize {
    STATE.with(|s| s.borrow().as_ref().unwrap().output.as_ptr() as usize)
}
#[no_mangle]
pub extern "C" fn output_len() -> usize {
    STATE.with(|s| s.borrow().as_ref().unwrap().output.len())
}
#[cfg(target_arch = "wasm32")]
mod local {
    #[link(wasm_import_module = "ic0")]
    extern "C" {
        fn performance_counter(kind: i32) -> i64;
        fn msg_arg_data_size() -> i32;
        fn msg_arg_data_copy(dst: i32, offset: i32, size: i32);
        fn msg_reply_data_append(src: i32, size: i32);
        fn msg_reply();
    }
    fn args<const N: usize>() -> [u8; N] {
        assert_eq!(unsafe { msg_arg_data_size() }, N as i32);
        let mut b = [0; N];
        unsafe { msg_arg_data_copy(b.as_mut_ptr() as i32, 0, N as i32) };
        b
    }
    fn reply(b: &[u8]) {
        unsafe {
            msg_reply_data_append(b.as_ptr() as i32, b.len() as i32);
            msg_reply();
        }
    }
    #[export_name = "canister_update configure"]
    pub extern "C" fn configure() {
        let b = args::<8>();
        super::setup(
            u32::from_le_bytes(b[..4].try_into().unwrap()) as usize,
            512,
            u32::from_le_bytes(b[4..].try_into().unwrap()) as usize,
        );
        super::run(0);
        super::run(1);
        reply(&[]);
    }
    #[export_name = "canister_query measure"]
    pub extern "C" fn measure() {
        let b = args::<8>();
        let backend = u32::from_le_bytes(b[..4].try_into().unwrap()) as usize;
        let begin = unsafe { performance_counter(0) } as u64;
        super::run(backend);
        let count = unsafe { performance_counter(0) } as u64 - begin;
        let mut hash = 14695981039346656037u64;
        super::STATE.with(|s| {
            for v in &s.borrow().as_ref().unwrap().output {
                for byte in v.to_le_bytes() {
                    hash = (hash ^ byte as u64).wrapping_mul(1099511628211);
                }
            }
        });
        let mut out = Vec::new();
        out.extend(count.to_le_bytes());
        out.extend(hash.to_le_bytes());
        out.extend((super::output_len() as u32).to_le_bytes());
        out.extend((core::arch::wasm32::memory_size(0) as u32).to_le_bytes());
        reply(&out);
    }
}
