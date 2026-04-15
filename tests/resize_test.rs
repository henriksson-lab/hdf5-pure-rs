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
fn test_resize_then_write_appended_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resize_write_chunk.h5");

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
        mf.resize_dataset("data", &[15]).unwrap();
        let chunk: Vec<i32> = (10..15).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<i32>(),
            )
        };
        mf.write_chunk("data", &[10], bytes).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let vals: Vec<i32> = f.dataset("data").unwrap().read::<i32>().unwrap();
        assert_eq!(vals, (0..15).collect::<Vec<_>>());
    }
}

#[test]
fn test_write_chunk_replaces_existing_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("replace_chunk.h5");

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
        let chunk: Vec<i32> = (100..105).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<i32>(),
            )
        };
        mf.write_chunk("data", &[5], bytes).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let vals: Vec<i32> = f.dataset("data").unwrap().read::<i32>().unwrap();
        assert_eq!(vals, vec![0, 1, 2, 3, 4, 100, 101, 102, 103, 104]);
    }
}

#[test]
fn test_write_chunk_splits_full_v1_btree_leaf() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("split_full_chunk_btree.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("data")
            .shape(&[320])
            .chunk(&[5])
            .resizable()
            .write::<i32>(&(0..320).collect::<Vec<_>>())
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("data", &[325]).unwrap();
        let chunk: Vec<i32> = (320..325).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<i32>(),
            )
        };
        mf.write_chunk("data", &[320], bytes).unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("data", &[330]).unwrap();
        let chunk: Vec<i32> = (325..330).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<i32>(),
            )
        };
        mf.write_chunk("data", &[325], bytes).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let vals: Vec<i32> = f.dataset("data").unwrap().read::<i32>().unwrap();
        assert_eq!(vals, (0..330).collect::<Vec<_>>());
    }
}

#[test]
fn test_write_chunk_replaces_existing_v4_fixed_array_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("replace_v4_fixed_array.h5");
    std::fs::copy("tests/data/hdf5_ref/v4_fixed_array_chunks.h5", &path).unwrap();

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        let chunk: Vec<i32> = (1000..1010).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<i32>(),
            )
        };
        mf.write_chunk("fixed_array", &[0], bytes).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let vals: Vec<i32> = f.dataset("fixed_array").unwrap().read::<i32>().unwrap();
        let mut expected: Vec<i32> = (0..100).collect();
        expected[..10].copy_from_slice(&(1000..1010).collect::<Vec<_>>());
        assert_eq!(vals, expected);
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
