use std::fs;
use std::path::Path;

use hdf5_pure_rust::{File, HyperslabDim, Selection};

const BDV_PATH: &str = "/big/henriksson/ome_images/BDV/idr0038/Lam-Cytok-MV-f0.h5";
const BDV_DATASET: &str = "t00000/s15/0/cells";
const BDV_CROP_PIXELS: usize = 16 * 16;
const BDV_RSS_DELTA_LIMIT_KIB: u64 = 64 * 1024;

const CELLH5_PATH: &str = "/big/henriksson/ome_images/CellH5/samples/full/0013.ch5";
const CELLH5_DATASET: &str =
    "/sample/0/plate/H2b_aTub_MD20x_exp911/experiment/0/position/0013/image/channel";
const CELLH5_PLANE_PIXELS: usize = 1040 * 1392;
const CELLH5_RSS_DELTA_LIMIT_KIB: u64 = 192 * 1024;
const CELLH5_METADATA_NODE_LIMIT: usize = 512;

fn peak_rss_kib() -> u64 {
    let status = fs::read_to_string("/proc/self/status")
        .expect("could not read /proc/self/status for VmHWM");
    status
        .lines()
        .find_map(|line| {
            let value = line.strip_prefix("VmHWM:")?;
            value
                .split_whitespace()
                .next()
                .and_then(|kib| kib.parse().ok())
        })
        .expect("could not find VmHWM in /proc/self/status")
}

fn skip_if_missing(path: &str) -> bool {
    if Path::new(path).exists() {
        false
    } else {
        eprintln!("skipping missing real-world benchmark fixture: {path}");
        true
    }
}

fn assert_rss_delta_under(before_kib: u64, limit_kib: u64, label: &str) {
    let after_kib = peak_rss_kib();
    let delta_kib = after_kib.saturating_sub(before_kib);
    eprintln!(
        "{label} peak RSS before={before_kib} KiB after={after_kib} KiB delta={delta_kib} KiB"
    );
    assert!(
        delta_kib <= limit_kib,
        "{label} RSS delta {delta_kib} KiB exceeds limit {limit_kib} KiB"
    );
}

fn walk_cellh5_metadata(file: &File, path: &str, visited: &mut usize) {
    if *visited >= CELLH5_METADATA_NODE_LIMIT {
        return;
    }
    *visited += 1;

    let Ok(group) = file.group(path) else {
        return;
    };
    let _ = group.attr_names();

    let mut members = Vec::new();
    if group
        .visit_member_names(|name| {
            members.push(name.to_string());
            Ok(())
        })
        .is_err()
    {
        return;
    }

    for member in members {
        if *visited >= CELLH5_METADATA_NODE_LIMIT {
            return;
        }
        let child_path = if path == "/" {
            format!("/{member}")
        } else {
            format!("{path}/{member}")
        };
        if let Ok(dataset) = file.dataset(&child_path) {
            *visited += 1;
            let _ = dataset.shape();
            let _ = dataset.dtype();
            let _ = dataset.attr_names();
        } else if file.group(&child_path).is_ok() {
            walk_cellh5_metadata(file, &child_path, visited);
        }
    }
}

#[test]
#[ignore = "reproduces the Bio-Formats BDV RSS regression on the local /big fixture"]
fn real_bdv_lam_cytok_first_plane_rss_regression() {
    if skip_if_missing(BDV_PATH) {
        return;
    }

    let file = File::open(BDV_PATH).expect("open BDV benchmark fixture");
    let dataset = file.dataset(BDV_DATASET).expect("open BDV cells dataset");
    let selection = Selection::Hyperslab(vec![
        HyperslabDim::new(0, 1, 1, 1),
        HyperslabDim::new(0, 1, 1, 16),
        HyperslabDim::new(0, 1, 1, 16),
    ]);
    let mut crop = vec![0u16; BDV_CROP_PIXELS];

    let before_kib = peak_rss_kib();
    dataset
        .read_slice_into::<u16, _>(selection, &mut crop)
        .expect("read BDV first full-resolution crop");

    assert_eq!(crop.len(), BDV_CROP_PIXELS);
    assert_rss_delta_under(
        before_kib,
        BDV_RSS_DELTA_LIMIT_KIB,
        "BDV Lam-Cytok first crop",
    );
}

#[test]
#[ignore = "reproduces the Bio-Formats CellH5 RSS regression on the local /big fixture"]
fn real_cellh5_0013_first_channel_plane_rss_regression() {
    if skip_if_missing(CELLH5_PATH) {
        return;
    }

    let before_kib = peak_rss_kib();
    let file = File::open(CELLH5_PATH).expect("open CellH5 benchmark fixture");
    let mut visited = 0;
    walk_cellh5_metadata(&file, "/", &mut visited);
    let dataset = file
        .dataset(CELLH5_DATASET)
        .expect("open CellH5 image/channel dataset");
    let selection = Selection::Hyperslab(vec![
        HyperslabDim::new(0, 1, 1, 1),
        HyperslabDim::new(0, 1, 1, 1),
        HyperslabDim::new(0, 1, 1, 1),
        HyperslabDim::new(0, 1, 1040, 1),
        HyperslabDim::new(0, 1, 1392, 1),
    ]);
    let mut plane = vec![0u8; CELLH5_PLANE_PIXELS];

    dataset
        .read_slice_into::<u8, _>(selection, &mut plane)
        .expect("read CellH5 first channel plane");

    assert_eq!(plane.len(), CELLH5_PLANE_PIXELS);
    assert_rss_delta_under(
        before_kib,
        CELLH5_RSS_DELTA_LIMIT_KIB,
        "CellH5 0013 first channel plane",
    );
}
