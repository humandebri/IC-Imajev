//! Query-local optional instruction profiling. Inclusive nested spans are named separately.
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
};
thread_local! {
    static CLOCK: Cell<Option<fn() -> u64>> = const { Cell::new(None) };
    static COUNTS: RefCell<BTreeMap<&'static str,(u64,u64)>> = const { RefCell::new(BTreeMap::new()) };
}
pub fn start(clock: fn() -> u64) {
    COUNTS.with(|c| c.borrow_mut().clear());
    CLOCK.with(|c| c.set(Some(clock)));
}
pub fn finish() -> Vec<(String, u64, u64)> {
    CLOCK.with(|c| c.set(None));
    COUNTS.with(|c| {
        c.borrow()
            .iter()
            .map(|(k, (instructions, calls))| (k.to_string(), *instructions, *calls))
            .collect()
    })
}
#[inline(always)]
pub fn measure<T>(name: &'static str, run: impl FnOnce() -> T) -> T {
    #[cfg(not(feature = "instruction-profile"))]
    {
        let _ = name;
        return run();
    }
    #[cfg(feature = "instruction-profile")]
    {
        let clock = CLOCK.with(|c| c.get());
        let Some(clock) = clock else {
            return run();
        };
        let before = clock();
        let result = run();
        let elapsed = clock() - before;
        COUNTS.with(|c| {
            let mut c = c.borrow_mut();
            let v = c.entry(name).or_default();
            v.0 += elapsed;
            v.1 += 1;
        });
        result
    }
}
