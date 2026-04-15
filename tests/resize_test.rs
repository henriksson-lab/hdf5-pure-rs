use hdf5_pure_rust::{File, MutableFile, WritableFile};

#[test]
fn test_resize_chunked_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resize_test.h5");

    // Create a chunked dataset with unlimited max dims
    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("data")
            .shape(&[10])
            .chunk(&[5])
            .resizable()
            .write::<f64>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
            .unwrap();
        wf.flush().unwrap();
    }

    // Verify initial shape
    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("data").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![10]);
        let vals: Vec<f64> = ds.read::<f64>().unwrap();
        assert_eq!(vals.len(), 10);
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[9], 10.0);
    }

    // Resize to smaller (shrink)
    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("data", &[7]).unwrap();
    }

    // Verify shrunk shape
    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("data").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![7]);
        let vals: Vec<f64> = ds.read::<f64>().unwrap();
        assert_eq!(vals.len(), 7);
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[6], 7.0);
    }

    // Resize to larger (extend -- new region reads as zeros)
    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("data", &[15]).unwrap();
    }

    // Verify extended shape
    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("data").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![15]);
        let vals: Vec<f64> = ds.read::<f64>().unwrap();
        assert_eq!(vals.len(), 15);
        assert_eq!(vals[0], 1.0);
        // Original data preserved in existing chunks
        assert_eq!(vals[4], 5.0);
    }
}

#[test]
fn test_resize_non_chunked_fails() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resize_nonchunked.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("contiguous")
            .write::<f64>(&[1.0, 2.0, 3.0])
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        let result = mf.resize_dataset("contiguous", &[5]);
        assert!(result.is_err());
    }
}

#[test]
fn test_resize_wrong_ndims_fails() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resize_wrongdims.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("data")
            .shape(&[10])
            .chunk(&[5])
            .resizable()
            .write::<i32>(&(0..10).collect::<Vec<_>>())
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        // Try 2D resize on 1D dataset
        let result = mf.resize_dataset("data", &[5, 2]);
        assert!(result.is_err());
    }
}

#[test]
fn test_mutable_file_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mutable_read.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("values")
            .write::<f32>(&[1.0, 2.0, 3.0])
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let mf = MutableFile::open_rw(&path).unwrap();
        let names = mf.member_names().unwrap();
        assert!(names.contains(&"values".to_string()));

        let ds = mf.dataset("values").unwrap();
        let vals: Vec<f32> = ds.read::<f32>().unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }
}
