#[cfg(feature = "tracehash")]
fn main() -> hdf5_pure_rust::Result<()> {
    hdf5_pure_rust::tracehash_corpus::walk_paths(["tests/data/datasets_v0.h5"].into_iter())
}

#[cfg(not(feature = "tracehash"))]
fn main() {
    eprintln!("run with --features tracehash");
}
