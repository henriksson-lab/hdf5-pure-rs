use hdf5_pure_rust::File;

fn main() -> hdf5_pure_rust::Result<()> {
    let file = File::open("tests/data/datasets_v0.h5")?;
    let dataset = file.dataset("float64_1d")?;
    let values: Vec<f64> = dataset.read()?;
    println!("{values:?}");
    Ok(())
}
