//! Standalone projection harness with optional local-only IC diagnostic exports.
use imajev_runtime::{
    int8_kernel::{quantize_rows, QuantizedRows},
    output_pairs::{project, PreparedPairs},
};
use std::cell::RefCell;

struct State {
    input: QuantizedRows,
    weights: PreparedPairs,
    scales: Vec<f32>,
    output: Vec<f32>,
}
thread_local! { static STATE: RefCell<Option<State>> = const { RefCell::new(None) }; }

fn value(index: usize, seed: u32) -> i8 {
    ((index as u32)
        .wrapping_mul(1_664_525)
        .wrapping_add(seed)
        .wrapping_shr(16)
        % 63) as i8
        - 31
}

#[no_mangle]
pub extern "C" fn setup(n: usize, rows: usize, cols: usize) {
    assert!(n > 0 && n <= 132 && rows == 512 && matches!(cols, 256 | 2560));
    let x: Vec<f32> = (0..n * cols)
        .map(|i| {
            if i % 32 == 0 {
                127.
            } else {
                value(i, 1_013_904_223) as f32
            }
        })
        .collect();
    let input = quantize_rows(&x, n, cols).unwrap();
    let mut weights: Vec<u8> = (0..rows * cols).map(|i| value(i, 777) as u8).collect();
    let scales = vec![1.; rows];
    for scale in &scales {
        weights.extend(f32::to_le_bytes(*scale));
    }
    let weights = PreparedPairs::from_le_bytes(&weights, rows, cols).unwrap();
    STATE.with(|s| {
        *s.borrow_mut() = Some(State {
            input,
            weights,
            scales,
            output: Vec::new(),
        })
    });
}

#[no_mangle]
pub extern "C" fn run() {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let s = s.as_mut().unwrap();
        s.output = project(&s.input, &s.weights.packed_rows(0, 512).unwrap(), &s.scales).unwrap();
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

#[cfg(all(feature = "local-ic", target_arch = "wasm32"))]
mod local_ic {
    #[link(wasm_import_module = "ic0")]
    extern "C" {
        fn performance_counter(kind: i32) -> i64;
        fn msg_arg_data_size() -> i32;
        fn msg_arg_data_copy(dst: i32, offset: i32, size: i32);
        fn msg_reply_data_append(src: i32, size: i32);
        fn msg_reply();
    }

    fn reply(bytes: &[u8]) {
        unsafe {
            msg_reply_data_append(bytes.as_ptr() as i32, bytes.len() as i32);
            msg_reply();
        }
    }

    #[export_name = "canister_update configure"]
    pub extern "C" fn configure() {
        assert_eq!(unsafe { msg_arg_data_size() }, 12);
        let mut args = [0u8; 12];
        unsafe {
            msg_arg_data_copy(args.as_mut_ptr() as i32, 0, 12);
        }
        let read = |i| u32::from_le_bytes(args[i..i + 4].try_into().unwrap()) as usize;
        super::setup(read(0), read(4), read(8));
        // Prime lazy operand capture in the persistent preparation update.
        super::run();
        reply(&[]);
    }

    #[export_name = "canister_query measure"]
    pub extern "C" fn measure() {
        let begin = unsafe { performance_counter(0) } as u64;
        super::run();
        let instructions = (unsafe { performance_counter(0) } as u64) - begin;
        let mut hash = 14_695_981_039_346_656_037u64;
        super::STATE.with(|s| {
            for value in &s.borrow().as_ref().unwrap().output {
                for byte in value.to_le_bytes() {
                    hash = (hash ^ byte as u64).wrapping_mul(1_099_511_628_211);
                }
            }
        });
        let mut bytes = Vec::new();
        bytes.extend(instructions.to_le_bytes());
        bytes.extend(hash.to_le_bytes());
        bytes.extend((super::output_len() as u32).to_le_bytes());
        bytes.extend((core::arch::wasm32::memory_size(0) as u32).to_le_bytes());
        reply(&bytes);
    }
}
