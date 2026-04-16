use hdf5_pure_rust::File;

fn open_real_world_fixture(path: &str) -> Option<File> {
    match File::open(path) {
        Ok(file) => Some(file),
        Err(err) => {
            eprintln!(
                "skipping optional real-world fixture {path}: {err}; run scripts/download-real-world-fixtures.py"
            );
            None
        }
    }
}

#[test]
fn test_real_world_anndata_h5ad_smoke() {
    let Some(f) = open_real_world_fixture("tests/data/real_world/anndataR_example.h5ad") else {
        return;
    };
    let members = f.member_names().unwrap();
    for expected in ["X", "layers", "obs", "obsm", "obsp", "uns", "var"] {
        assert!(
            members.contains(&expected.to_string()),
            "missing {expected}"
        );
    }

    let x_data: Vec<f32> = f.dataset("X/data").unwrap().read::<f32>().unwrap();
    let x_indices: Vec<i32> = f.dataset("X/indices").unwrap().read::<i32>().unwrap();
    let x_indptr: Vec<i32> = f.dataset("X/indptr").unwrap().read::<i32>().unwrap();
    assert_eq!(x_data.len(), 4317);
    assert_eq!(x_indices.len(), 4317);
    assert_eq!(x_indptr.len(), 51);
    assert_eq!(x_indptr[0], 0);
    assert_eq!(*x_indptr.last().unwrap(), 4317);

    let dense_x: Vec<f32> = f.dataset("layers/dense_X").unwrap().read::<f32>().unwrap();
    assert_eq!(dense_x.len(), 50 * 100);

    let obs_index = f.dataset("obs/_index").unwrap().read_strings().unwrap();
    let var_index = f.dataset("var/_index").unwrap().read_strings().unwrap();
    assert_eq!(obs_index.len(), 50);
    assert_eq!(var_index.len(), 100);

    let pca: Vec<f32> = f.dataset("obsm/X_pca").unwrap().read::<f32>().unwrap();
    assert_eq!(pca.len(), 50 * 38);
}

#[test]
fn test_real_world_keras_h5_model_smoke() {
    let Some(f) = open_real_world_fixture("tests/data/real_world/keras_conv_mnist_tf_model.h5")
    else {
        return;
    };
    let members = f.member_names().unwrap();
    assert!(members.contains(&"model_weights".to_string()));
    assert!(members.contains(&"optimizer_weights".to_string()));

    let conv_kernel: Vec<f32> = f
        .dataset("model_weights/conv2d_2/conv2d_2/kernel:0")
        .unwrap()
        .read::<f32>()
        .unwrap();
    let conv_bias: Vec<f32> = f
        .dataset("model_weights/conv2d_2/conv2d_2/bias:0")
        .unwrap()
        .read::<f32>()
        .unwrap();
    let dense_kernel: Vec<f32> = f
        .dataset("model_weights/dense_1/dense_1/kernel:0")
        .unwrap()
        .read::<f32>()
        .unwrap();
    let dense_bias: Vec<f32> = f
        .dataset("model_weights/dense_1/dense_1/bias:0")
        .unwrap()
        .read::<f32>()
        .unwrap();

    assert_eq!(conv_kernel.len(), 3 * 3 * 1 * 32);
    assert_eq!(conv_bias.len(), 32);
    assert_eq!(dense_kernel.len(), 1600 * 10);
    assert_eq!(dense_bias.len(), 10);
    assert!(conv_kernel.iter().any(|v| *v != 0.0));
    assert!(dense_kernel.iter().any(|v| *v != 0.0));
}

#[test]
fn test_real_world_h5py_smoke() {
    let Some(f) = open_real_world_fixture("tests/data/real_world/h5py_3_12_smoke.h5") else {
        return;
    };
    let run = f.group("experiment/run_001").unwrap();
    assert_eq!(
        run.attr("temperature_c").unwrap().read_scalar_f64(),
        Some(21.5)
    );

    let image_stack: Vec<u16> = f
        .dataset("experiment/run_001/image_stack")
        .unwrap()
        .read::<u16>()
        .unwrap();
    assert_eq!(image_stack, (0u16..24).collect::<Vec<_>>());

    let signal: Vec<f64> = f
        .dataset("experiment/run_001/signal")
        .unwrap()
        .read::<f64>()
        .unwrap();
    assert_eq!(signal.len(), 25);
    assert!((signal[0] - 0.0).abs() < 1e-12);
    assert!((signal[24] - 1.0).abs() < 1e-12);

    let labels = f
        .dataset("experiment/run_001/labels")
        .unwrap()
        .read_strings()
        .unwrap();
    assert_eq!(labels, vec!["alpha", "βeta", "猫"]);

    let table = f.dataset("experiment/run_001/compound_table").unwrap();
    let fields = table.compound_fields().unwrap();
    assert_eq!(
        fields.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(),
        vec!["id", "score"]
    );
    assert_eq!(table.read_field::<i32>("id").unwrap(), vec![1, 2, 3]);
    assert_eq!(
        table.read_field::<f64>("score").unwrap(),
        vec![0.5, 0.75, 1.25]
    );
}

#[test]
fn test_real_world_10x_feature_barcode_matrix_smoke() {
    let Some(f) = open_real_world_fixture(
        "tests/data/real_world/10x_pbmc_1k_v3_filtered_feature_bc_matrix.h5",
    ) else {
        return;
    };

    let members = f.member_names().unwrap();
    assert!(members.contains(&"matrix".to_string()));

    let data: Vec<i32> = f.dataset("matrix/data").unwrap().read::<i32>().unwrap();
    let indices: Vec<i32> = f.dataset("matrix/indices").unwrap().read::<i32>().unwrap();
    let indptr: Vec<i32> = f.dataset("matrix/indptr").unwrap().read::<i32>().unwrap();
    let shape: Vec<i32> = f.dataset("matrix/shape").unwrap().read::<i32>().unwrap();
    let barcodes = f
        .dataset("matrix/barcodes")
        .unwrap()
        .read_strings()
        .unwrap();
    let feature_ids = f
        .dataset("matrix/features/id")
        .unwrap()
        .read_strings()
        .unwrap();

    assert_eq!(data.len(), indices.len());
    assert_eq!(shape.len(), 2);
    assert_eq!(barcodes.len(), shape[1] as usize);
    assert_eq!(feature_ids.len(), shape[0] as usize);
    assert_eq!(indptr.len(), barcodes.len() + 1);
    assert_eq!(indptr[0], 0);
    assert_eq!(*indptr.last().unwrap(), data.len() as i32);
}

#[test]
fn test_real_world_netcdf4_like_smoke() {
    let Some(f) = open_real_world_fixture("tests/data/real_world/netcdf4_like_climate.nc") else {
        return;
    };

    let lat: Vec<f32> = f.dataset("lat").unwrap().read::<f32>().unwrap();
    let lon: Vec<f32> = f.dataset("lon").unwrap().read::<f32>().unwrap();
    let temperature = f.dataset("temperature").unwrap();
    let values: Vec<f32> = temperature.read::<f32>().unwrap();

    assert_eq!(lat, vec![-45.0, 0.0, 45.0]);
    assert_eq!(lon, vec![0.0, 90.0, 180.0, 270.0]);
    assert_eq!(temperature.shape().unwrap(), vec![3, 4]);
    assert_eq!(values.len(), 12);
    assert!((values[0] - 273.15).abs() < 1e-4);
}

#[test]
fn test_real_world_matlab_v73_like_smoke() {
    let Some(f) = open_real_world_fixture("tests/data/real_world/matlab_v73_like.mat") else {
        return;
    };

    let a: Vec<f64> = f.dataset("A").unwrap().read::<f64>().unwrap();
    let name: Vec<u16> = f.dataset("name").unwrap().read::<u16>().unwrap();
    let cell_refs = f.dataset("cell").unwrap();

    assert_eq!(a, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(name, "hello".encode_utf16().collect::<Vec<_>>());
    assert_eq!(cell_refs.shape().unwrap(), vec![1]);
}

#[test]
fn test_real_world_nexus_smoke() {
    let Some(f) = open_real_world_fixture("tests/data/real_world/nexus_simple.nxs") else {
        return;
    };

    let members = f.member_names().unwrap();
    assert!(members.contains(&"entry".to_string()));
    let counts: Vec<i32> = f
        .dataset("entry/instrument/detector/counts")
        .unwrap()
        .read::<i32>()
        .unwrap();
    assert_eq!(counts, (0..12).collect::<Vec<_>>());
}

#[test]
fn test_real_world_pandas_hdfstore_smoke() {
    let Some(f) = open_real_world_fixture("tests/data/real_world/pandas_hdfstore_table.h5") else {
        return;
    };

    let observations = f.group("observations").unwrap();
    let members = observations.member_names().unwrap();
    assert!(members.contains(&"table".to_string()));
    let table = f.dataset("observations/table").unwrap();
    assert_eq!(table.shape().unwrap()[0], 4);
}
