//! Immutable finite F32 weights, decoded once during fixed-model preparation.
use crate::Result;
#[cfg(feature = "experimental-f32-output-reuse")]
use std::cell::OnceCell;
use std::ops::Deref;
use std::rc::Rc;

#[derive(Clone)]
pub struct PreparedF32 {
    values: Rc<[f32]>,
    start: usize,
    len: usize,
    #[cfg(feature = "experimental-f32-output-reuse")]
    output_rows: Option<usize>,
    #[cfg(feature = "experimental-f32-output-reuse")]
    output_cols: usize,
    #[cfg(feature = "experimental-f32-output-reuse")]
    original: Rc<OnceCell<Vec<f32>>>,
}
impl PreparedF32 {
    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() % 4 != 0 {
            return Err("prepared f32 length".into());
        }
        let values: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if !values.iter().all(|v| v.is_finite()) {
            return Err("nonfinite prepared weight".into());
        }
        let len = values.len();
        Ok(Self {
            values: values.into(),
            start: 0,
            len,
            #[cfg(feature = "experimental-f32-output-reuse")]
            output_rows: None,
            #[cfg(feature = "experimental-f32-output-reuse")]
            output_cols: 64,
            #[cfg(feature = "experimental-f32-output-reuse")]
            original: Rc::new(OnceCell::new()),
        })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[cfg(all(test,feature="experimental-f32-output-reuse"))]
    pub(crate) fn original_materialized(&self)->bool {self.original.get().is_some()}
    #[cfg(feature = "experimental-f32-output-reuse")]
    pub fn from_le_bytes_output64(bytes: &[u8], rows: usize) -> Result<Self> {
        Self::from_le_bytes_output(bytes, rows, 64)
    }
    #[cfg(feature = "experimental-f32-output-reuse")]
    pub fn from_le_bytes_output(bytes: &[u8], rows: usize, cols: usize) -> Result<Self> {
        let original = Self::from_le_bytes(bytes)?;
        if rows == 0
            || rows % 32 != 0
            || cols == 0
            || cols % 64 != 0
            || cols > 9216
            || (cols != 64 && !cfg!(feature="experimental-f32-output-generic"))
            || rows.checked_mul(cols) != Some(original.len)
        {
            return Err("prepared output F32 shape".into());
        }
        let mut values = Vec::with_capacity(original.len);
        for tile in (0..rows).step_by(32) {
            for c in 0..cols {
                for r in 0..32 {
                    values.push(original[tile * cols + r * cols + c]);
                }
            }
        }
        Ok(Self {
            values: values.into(),
            start: 0,
            len: original.len,
            output_rows: Some(rows),
            output_cols: cols,
            original: Rc::new(OnceCell::new()),
        })
    }
    #[cfg(feature = "experimental-f32-output-reuse")]
    pub(crate) fn output_view(&self) -> Option<(&[f32], usize, usize)> {
        self.output_rows?;
        if self.start % self.output_cols != 0 || self.len % self.output_cols != 0 {
            return None;
        }
        Some((
            &self.values,
            self.start / self.output_cols,
            self.output_cols,
        ))
    }
    pub fn slice(&self, start: usize, len: usize) -> Option<Self> {
        if start.checked_add(len)? > self.len {
            return None;
        }
        Some(Self {
            values: self.values.clone(),
            start: self.start + start,
            len,
            #[cfg(feature = "experimental-f32-output-reuse")]
            output_rows: self.output_rows,
            #[cfg(feature = "experimental-f32-output-reuse")]
            output_cols: self.output_cols,
            #[cfg(feature = "experimental-f32-output-reuse")]
            original: Rc::new(OnceCell::new()),
        })
    }
}
impl Deref for PreparedF32 {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        #[cfg(feature = "experimental-f32-output-reuse")]
        if self.output_rows.is_some() {
            return self.original.get_or_init(|| {
                (self.start..self.start + self.len)
                    .map(|i| {
                        let row = i / self.output_cols;
                        let c = i % self.output_cols;
                        self.values[(row / 32) * 32 * self.output_cols + c * 32 + row % 32]
                    })
                    .collect()
            });
        }
        &self.values[self.start..self.start + self.len]
    }
}

/// An optional prepared view can only be created through finite validation.
pub trait WeightBuffer: AsRef<[u8]> {
    fn prepared_f32(&self) -> Option<PreparedF32> {
        None
    }
    #[cfg(feature = "experimental-prepared-output-pairs")]
    fn prepared_output_pairs(&self) -> Option<crate::output_pairs::PackedView> {
        None
    }
}
impl WeightBuffer for Vec<u8> {}
pub(crate) struct ByteOnly<B>(pub B);
impl<B: AsRef<[u8]>> AsRef<[u8]> for ByteOnly<B> {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}
impl<B: AsRef<[u8]>> WeightBuffer for ByteOnly<B> {}

pub(crate) enum LoadedWeight {
    Owned(Vec<f32>),
    Prepared(PreparedF32),
}
impl LoadedWeight {
    pub fn into_owned(self) -> Vec<f32> {
        match self {
            Self::Owned(v) => v,
            Self::Prepared(v) => v.to_vec(),
        }
    }
}
impl Deref for LoadedWeight {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        match self {
            Self::Owned(v) => v,
            Self::Prepared(v) => v,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature="experimental-f32-output-generic")]
    #[test]
    fn generic_columns_views_and_sum_order_never_inverse_decode_on_projection() {
        for cols in [128,2560,9216] {
            let rows=64;let weights:Vec<f32>=(0..rows*cols).map(|i|[-0.,0.,1.,-1.,0.1234567,f32::MIN_POSITIVE][i%6]).collect();
            let bytes:Vec<u8>=weights.iter().flat_map(|v|v.to_le_bytes()).collect();
            let fixed=PreparedF32::from_le_bytes_output(&bytes,rows,cols).unwrap();
            for (n,start,count) in [(1,0,64),(7,2,8),(45,31,33),(87,32,32)] {
                let view=fixed.slice(start*cols,count*cols).unwrap();
                let x:Vec<f32>=(0..n*cols).map(|i|[-0.,0.,1.,-1.,0.2345678,-0.99999994][i%6]).collect();
                let actual=crate::f32_output::project(&x,&view,n,count,cols).unwrap().unwrap();
                assert!(view.original.get().is_none());
                let expected=crate::matrix_reference(&x,&weights[start*cols..(start+count)*cols],n,count,cols).unwrap();
                assert!(actual.iter().zip(expected).all(|(a,b)|a.to_bits()==b.to_bits()));
                assert!(view.iter().zip(&weights[start*cols..(start+count)*cols]).all(|(a,b)|a.to_bits()==b.to_bits()));
            }
            assert!(fixed.original.get().is_none());
        }
    }
    #[cfg(feature = "experimental-f32-output-reuse")]
    #[test]
    fn output64_keeps_original_ranges_and_projects_without_inverse_decode() {
        let rows = 256;
        let cols = 64;
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| [-0., 0., 0.1234567, -0.99999994, 1., -1., f32::MIN_POSITIVE][i % 7])
            .collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let fixed = PreparedF32::from_le_bytes_output64(&bytes, rows).unwrap();
        assert_eq!(fixed.len(), values.len());
        assert!(fixed.original.get().is_none());
        for (n, start, count) in [
            (1, 0, 32),
            (7, 2, 8),
            (45, 24, 64),
            (80, 31, 33),
            (87, 32, 224),
            (89, 249, 7),
            (132, 0, 128),
        ] {
            let view = fixed.slice(start * 64, count * 64).unwrap();
            let x: Vec<f32> = (0..n * cols)
                .map(|i| [-0., 0., 1., -1., 0.2345678, f32::MIN_POSITIVE][i % 6])
                .collect();
            let actual = crate::f32_output::project(&x, &view, n, count, cols)
                .unwrap()
                .unwrap();
            assert!(view.original.get().is_none());
            let expected = crate::matrix_reference(
                &x,
                &values[start * 64..(start + count) * 64],
                n,
                count,
                cols,
            )
            .unwrap();
            assert!(actual
                .iter()
                .zip(expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
            assert!(view
                .iter()
                .zip(&values[start * 64..(start + count) * 64])
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        for (start, count) in [(1, 31), (63, 513), (values.len() - 3, 3)] {
            let view = fixed.slice(start, count).unwrap();
            assert!(view
                .iter()
                .zip(&values[start..start + count])
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        assert!(fixed.original.get().is_none());
        assert!(PreparedF32::from_le_bytes_output64(&bytes, rows - 1).is_err());
        assert!(PreparedF32::from_le_bytes_output64(&[], usize::MAX).is_err());
    }
    #[test]
    fn prepared_views_preserve_bits_ranges_and_lifetime() {
        let x = [0., -0., f32::from_bits(1), f32::MAX, -3.1415927];
        let bytes: Vec<_> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let prepared = PreparedF32::from_le_bytes(&bytes).unwrap();
        let view = prepared.slice(1, 3).unwrap();
        assert_eq!(view.as_ptr(), prepared[1..].as_ptr());
        assert!(prepared.slice(5, 1).is_none());
        assert!(prepared.slice(usize::MAX, 1).is_none());
        drop(prepared);
        assert_eq!(
            view.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            x[1..4].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert!(PreparedF32::from_le_bytes(&[0; 3]).is_err());
        for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(PreparedF32::from_le_bytes(&v.to_le_bytes()).is_err());
        }
    }
}
