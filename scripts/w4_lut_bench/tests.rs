#[path="src/kernel.rs"] mod kernel;
#[test]
fn layout_across_output_tiles_and_block_boundaries() {
    let rows=48;let cols=768;
    for n in [1,3,8,9] {
        let q:Vec<i16>=(0..n*cols).map(|i|((i*37+i/cols*19)%255)as i16-127).collect();
        let w:Vec<i8>=(0..rows*cols).map(|i|((i*7+i/cols*3+i/16)%16)as i8-8).collect();
        let sx:Vec<f32>=(0..n*3).map(|i|0.001*(i+1)as f32).collect();
        let sw:Vec<f32>=(0..rows*3).map(|i|0.002*(i%11+1)as f32).collect();
        let tab=kernel::tables(&q);let packed=kernel::pack(&w,rows,cols);
        let dense=kernel::dense(&q,&sx,&w,&sw,n,rows,cols);
        let lut=kernel::lut(&q,&sx,&packed,&sw,n,rows,cols,&tab);
        assert!(dense.iter().zip(&lut).all(|(a,b)|a.to_bits()==b.to_bits()));
    }
}
#[test]
fn activation_rounding_and_nonfinite_rejection() {
    let mut x=vec![0.;256];x[..6].copy_from_slice(&[127.,-127.,0.5,1.5,2.5,-2.5]);
    let (q,s)=kernel::quantize(&x,256);assert_eq!(s,[1.]);assert_eq!(&q[..6],&[127,-127,0,2,2,-2]);
    x[255]=f32::NAN;assert!(std::panic::catch_unwind(||kernel::quantize(&x,256)).is_err());
}
