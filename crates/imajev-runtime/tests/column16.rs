#![cfg(feature="experimental-column16")]
use imajev_runtime::int8_kernel::{quantize_rows,project_column16};
#[test]
fn column16_matches_scalar_blocks_for_extremes_and_token_tails() {
    for n in [1,7,8,31,32,33,45,47,48,49,64,81,82,83,84,85,86,87,88,89,95,96,97,132] {
        for cols in [256,512] {
            let rows=32;
            let x:Vec<_>=(0..n*cols).map(|i|[-127.,127.,0.,-1.,1.,0.5][i%6]).collect();
            let w:Vec<_>=(0..rows*cols).map(|i|[-128i8,127,-127,0,1,-1][i%6]).collect();
            let sw:Vec<_>=(0..rows).map(|i|0.0123*(i+1)as f32).collect();
            let q=quantize_rows(&x,n,cols).unwrap();
            let got=project_column16(&q,&w,&sw,rows).unwrap();
            #[cfg(feature="experimental-adopt-column16-token48")]
            {
                let adopted=imajev_runtime::int8_kernel::project(&q,&w,&sw,rows).unwrap();
                assert!(got.iter().zip(&adopted).all(|(a,b)|a.to_bits()==b.to_bits()), "adopted tile n={n},cols={cols}");
            }
            #[cfg(feature="experimental-column16-token48")]
            {
                let wide=imajev_runtime::int8_kernel::project_column16_token48(&q,&w,&sw,rows).unwrap();
                assert!(got.iter().zip(&wide).all(|(a,b)|a.to_bits()==b.to_bits()), "48-token tile n={n},cols={cols}");
            }
            #[cfg(feature="experimental-dot-scale")]
            {
                let fused=imajev_runtime::int8_kernel::project_dot_scale(&q,&w,&sw,rows).unwrap();
                assert!(got.iter().zip(&fused).all(|(a,b)|a.to_bits()==b.to_bits()), "dot-scale API n={n},cols={cols}");
            }
            #[cfg(feature="experimental-balanced44")]
            {
                let balanced=imajev_runtime::int8_kernel::project_balanced44(&q,&w,&sw,rows).unwrap();
                assert!(got.iter().zip(&balanced).all(|(a,b)|a.to_bits()==b.to_bits()), "balanced n={n},cols={cols}");
            }
            #[cfg(feature="experimental-column32")]
            {
                let wide=imajev_runtime::int8_kernel::project_column32_balanced(&q,&w,&sw,rows).unwrap();
                assert!(got.iter().zip(&wide).all(|(a,b)|a.to_bits()==b.to_bits()), "column32 n={n},cols={cols}");
            }
            for t in 0..n { for r in 0..rows {
                let mut expected=0f32;
                for b in 0..cols/256 {
                    let mut dot=0i32;
                    for k in b*256..(b+1)*256 {dot+=q.values()[t*cols+k]as i32*w[r*cols+k]as i32;}
                    expected+=(dot as f32*q.scales()[t*(cols/256)+b])*sw[r];
                }
                assert_eq!(got[t*rows+r].to_bits(),expected.to_bits(),"n={n},cols={cols},t={t},r={r}");
            }}
        }
    }
}
#[test]
fn column16_rejects_incomplete_output_tiles_and_invalid_scales() {
    let q=quantize_rows(&vec![0.;256],1,256).unwrap();
    for rows in [0,8,24] {
        assert!(project_column16(&q,&vec![0;rows*256],&vec![1.;rows],rows).is_err());
        #[cfg(feature="experimental-column16-token48")]
        assert!(imajev_runtime::int8_kernel::project_column16_token48(&q,&vec![0;rows*256],&vec![1.;rows],rows).is_err());
    }
    for bad in [0.,-1.,f32::NAN,f32::INFINITY] {
        let mut s=vec![1.;16];s[15]=bad;
        assert!(project_column16(&q,&vec![0;16*256],&s,16).is_err());
        #[cfg(feature="experimental-column16-token48")]
        assert!(imajev_runtime::int8_kernel::project_column16_token48(&q,&vec![0;16*256],&s,16).is_err());
    }
}

#[cfg(feature="experimental-column32")]
#[test]
fn column32_rejects_invalid_weights_in_the_selected_shape() {
    let q=quantize_rows(&vec![1.;87*256],87,256).unwrap();
    assert!(imajev_runtime::int8_kernel::project_column32_balanced(&q,&vec![0;32*256-1],&vec![1.;32],32).is_err());
    for bad in [0.,-1.,f32::NAN,f32::INFINITY] {
        let mut s=vec![1.;32];s[31]=bad;
        assert!(imajev_runtime::int8_kernel::project_column32_balanced(&q,&vec![0;32*256],&s,32).is_err());
    }
}
