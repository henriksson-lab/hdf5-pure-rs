fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: tracehash_corpus <file.h5>...");
        std::process::exit(2);
    }

    let mut failures = 0;
    if let Err(err) = hdf5_pure_rust::tracehash_corpus::walk_paths(paths.iter().map(String::as_str))
    {
        eprintln!("{err}");
        failures += 1;
    }

    if failures != 0 {
        std::process::exit(1);
    }
}
