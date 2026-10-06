//! Local diagnostic only. Laya sources are copied into the frozen build directory.
use std::cell::RefCell;
#[macro_export]
macro_rules! bail {
    ($e:expr) => {
        return Err($e)
    };
}
mod candle_core {
    pub type Result<T> = std::result::Result<T, &'static str>;
    pub use crate::bail;
}
mod baseline {
    use crate::candle_core;
    include!(concat!(env!("LAYA_PEAK_SOURCE"), "/baseline.rs"));
}
mod candidate {
    use crate::candle_core;
    include!(concat!(env!("LAYA_PEAK_SOURCE"), "/candidate.rs"));
}
struct State {
    n: usize,
    cols: usize,
    input: Vec<f32>,
    output: Vec<u8>,
    valid: bool,
}
thread_local! {static STATE:RefCell<Option<State>>=const{RefCell::new(None)};}
#[no_mangle]
pub extern "C" fn setup(n: usize, cols: usize, pattern: usize) {
    assert!(n > 0 && n <= 128 && cols > 0 && cols <= 16384 && pattern <= 8);
    let edges = [
        127.,
        -127.,
        0.,
        -0.,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        f32::from_bits(0x3effffff),
        f32::from_bits(0x3f000001),
    ];
    let mut input: Vec<f32> = (0..n * cols)
        .map(|i| match pattern {
            1 => edges[i % edges.len()],
            2 => {
                if i % 2 == 0 {
                    0.
                } else {
                    -0.
                }
            }
            3 => f32::from_bits((i % 127 + 1) as u32 | if i % 2 == 0 { 0 } else { 0x80000000 }),
            4 => {
                if i % 2 == 0 {
                    f32::MAX
                } else {
                    -f32::MAX
                }
            }
            _ => ((i * 37 % 211) as f32 - 105.) / 31.,
        })
        .collect();
    if pattern >= 5 {
        *input.last_mut().unwrap() = f32::from_bits(match pattern {
            5 => 0x7f800000,
            6 => 0x7fc12345,
            7 => 0x7f800001,
            _ => 0xffc12345,
        });
    }
    STATE.with(|s| {
        *s.borrow_mut() = Some(State {
            n,
            cols,
            input,
            output: Vec::new(),
            valid: false,
        })
    });
}
#[no_mangle]
pub extern "C" fn run(backend: usize) -> usize {
    assert!(backend <= 1);
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let s = state.as_mut().unwrap();
        let mut q = vec![0i8; s.input.len()];
        let mut scales = vec![1f32; s.n];
        let mut valid = true;
        for r in 0..s.n {
            let input = &s.input[r * s.cols..(r + 1) * s.cols];
            let out = &mut q[r * s.cols..(r + 1) * s.cols];
            let scale = if backend == 0 {
                baseline::row(input, out)
            } else {
                candidate::row(input, out)
            };
            match scale {
                Ok(scale) => scales[r] = scale,
                Err(_) => {
                    valid = false;
                    break;
                }
            }
        }
        let mut bytes = Vec::with_capacity(scales.len() * 4 + q.len());
        for scale in scales {
            bytes.extend(scale.to_le_bytes());
        }
        bytes.extend(q.iter().map(|v| *v as u8));
        s.output = bytes;
        s.valid = valid;
        valid as usize
    })
}
#[no_mangle]
pub extern "C" fn input_ptr() -> usize {
    STATE.with(|s| s.borrow().as_ref().unwrap().input.as_ptr() as usize)
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
mod ic {
    #[link(wasm_import_module = "ic0")]
    extern "C" {
        fn performance_counter(k: i32) -> i64;
        fn msg_arg_data_size() -> i32;
        fn msg_arg_data_copy(d: i32, o: i32, n: i32);
        fn msg_reply_data_append(p: i32, n: i32);
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
        let b = args::<12>();
        let v = |i| u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
        super::setup(v(0), v(4), v(8));
        super::run(0);
        super::run(1);
        reply(&[]);
    }
    #[export_name = "canister_query measure"]
    pub extern "C" fn measure() {
        let b = args::<8>();
        let backend = u32::from_le_bytes(b[..4].try_into().unwrap()) as usize;
        let start = unsafe { performance_counter(0) } as u64;
        let valid = super::run(backend);
        let count = unsafe { performance_counter(0) } as u64 - start;
        let mut hash = 14695981039346656037u64;
        super::STATE.with(|s| {
            for byte in &s.borrow().as_ref().unwrap().output {
                hash = (hash ^ *byte as u64).wrapping_mul(1099511628211);
            }
        });
        let mut out = Vec::new();
        out.extend(count.to_le_bytes());
        out.extend(hash.to_le_bytes());
        out.extend((valid as u32).to_le_bytes());
        out.extend((super::output_len() as u32).to_le_bytes());
        reply(&out);
    }
}
