use hdf5_pure_rust::{File, WritableFile};
use std::time::Instant;

fn main() {
    let n = 1_000_000usize;
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let iterations = 10;

    // Write benchmark
    let mut total_write = std::time::Duration::ZERO;
    for _ in 0..iterations {
        let t = Instant::now();
        let mut wf = WritableFile::create("/tmp/bench_rust.h5").unwrap();
        wf.new_dataset_builder("data")
            .shape(&[n as u64])
            .chunk(&[50000])
            .deflate(1)
            .write::<f64>(&data)
            .unwrap();
        wf.flush().unwrap();
        total_write += t.elapsed();
    }
    let avg_write = total_write / iterations as u32;

    // Read benchmark
    let mut total_read = std::time::Duration::ZERO;
    for _ in 0..iterations {
        let t = Instant::now();
        let f = File::open("/tmp/bench_rust.h5").unwrap();
        let ds = f.dataset("data").unwrap();
        let _vals: Vec<f64> = ds.read::<f64>().unwrap();
        total_read += t.elapsed();
    }
    let avg_read = total_read / iterations as u32;

    println!("hdf5-pure-rust write: {:.1} ms", avg_write.as_secs_f64() * 1000.0);
    println!("hdf5-pure-rust read:  {:.1} ms", avg_read.as_secs_f64() * 1000.0);
    println!("Data: {} f64 elements, chunked 50000, deflate level 1", n);

    std::fs::remove_file("/tmp/bench_rust.h5").ok();
}
