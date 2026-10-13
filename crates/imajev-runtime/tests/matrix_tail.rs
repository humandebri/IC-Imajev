#![cfg(feature="experimental-matrix-tail")]
use imajev_runtime::{matrix_tail,matrix_reference};
#[test]
fn groups_and_padded_lanes_match_scalar_column_order() {
    for n in [1,3,4,5,7,8,15,16,17,31,32,33,45,63,64,65,80,87,89,127,128,132] {
        for rows in [1,15,16,17,64] {for cols in [1,7,64] {
            let x:Vec<_>=(0..n*cols).map(|i|[0.,-0.,0.125,-3.75,f32::from_bits(1),16.25][i%6]).collect();
            let w:Vec<_>=(0..rows*cols).map(|i|[-0.,0.125,-0.75,4.5,f32::from_bits(1)][i%5]).collect();
            let old=matrix_reference(&x,&w,n,rows,cols).unwrap();
            let new=matrix_tail(&x,&w,n,rows,cols).unwrap();
            #[cfg(feature="experimental-adopt-matrix-tail")]
            {
                let adopted=imajev_runtime::matrix(&x,&w,n,rows,cols).unwrap();
                assert!(old.iter().zip(&adopted).all(|(a,b)|a.to_bits()==b.to_bits()),"selected n={n} rows={rows} cols={cols}");
            }
            assert_eq!(old.len(),new.len());
            assert!(old.iter().zip(new).all(|(a,b)|a.to_bits()==b.to_bits()),"n={n} rows={rows} cols={cols}");
        }}
    }
}
#[test]
fn malformed_shapes_fail_before_allocation() {
    for (n,r,c) in [(0,1,1),(1,0,1),(1,1,0),(usize::MAX,2,2),(1,900001,1)] {
        assert!(matrix_tail(&[],&[],n,r,c).is_err());
    }
    assert!(matrix_tail(&[0.;4],&[0.;4],4,2,1).is_err());
}
