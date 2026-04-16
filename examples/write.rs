use hdf5_pure_rust::WritableFile;

fn main() -> hdf5_pure_rust::Result<()> {
    let mut file = WritableFile::create("example-write.h5")?;
    file.new_dataset_builder("values")
        .shape(&[4])
        .fill_value::<i32>(-1)
        .write::<i32>(&[1, 2, 3, 4])?;
    file.flush()?;
    Ok(())
}
