use hdf5_pure_rust::File;

fn main() -> hdf5_pure_rust::Result<()> {
    let file = File::open("tests/data/hdf5_ref/vds_all.h5")?;
    let dataset = file.dataset("vds_all")?;
    let values: Vec<i32> = dataset.read()?;
    println!("{values:?}");
    Ok(())
}
