use std::fs;
use std::process::Command;

#[test]
fn invalid_h5type_derives_emit_compile_errors() {
    let cases = [
        (
            "union_type",
            r#"
use hdf5_pure_rust::DeriveH5Type;

#[derive(Copy, Clone, DeriveH5Type)]
union Bad {
    value: u8,
}
"#,
            "cannot derive `H5Type` for unions",
        ),
        (
            "unit_struct",
            r#"
use hdf5_pure_rust::DeriveH5Type;

#[derive(Copy, Clone, DeriveH5Type)]
struct Bad;
"#,
            "cannot derive `H5Type` for unit structs",
        ),
        (
            "missing_repr",
            r#"
use hdf5_pure_rust::DeriveH5Type;

#[derive(Copy, Clone, DeriveH5Type)]
struct Bad {
    value: u8,
}
"#,
            "`H5Type` requires #[repr(C)], #[repr(packed)], or #[repr(transparent)] for structs",
        ),
        (
            "enum_missing_repr",
            r#"
use hdf5_pure_rust::DeriveH5Type;

#[derive(Copy, Clone, DeriveH5Type)]
enum Bad {
    Value = 1,
}
"#,
            "`H5Type` requires explicit integer repr for enums",
        ),
        (
            "enum_data_variant",
            r#"
use hdf5_pure_rust::DeriveH5Type;

#[derive(Copy, Clone, DeriveH5Type)]
#[repr(u8)]
enum Bad {
    Value(u8) = 1,
}
"#,
            "`H5Type` can only be derived for enums with scalar discriminants",
        ),
    ];

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("Cargo.toml"),
        format!(
            r#"[package]
name = "derive-compile-fail"
version = "0.0.0"
edition = "2021"

[dependencies]
hdf5-pure-rust = {{ path = "{manifest_dir}" }}
"#
        ),
    )
    .unwrap();
    fs::create_dir(dir.path().join("src")).unwrap();
    let mut lib = String::new();
    for (name, source, _) in &cases {
        lib.push_str("#[allow(dead_code)]\nmod ");
        lib.push_str(name);
        lib.push_str(" {\n");
        lib.push_str(source);
        lib.push_str("\n}\n");
    }
    fs::write(dir.path().join("src/lib.rs"), lib).unwrap();

    let output = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string()))
        .arg("check")
        .arg("--quiet")
        .env("CARGO_TARGET_DIR", dir.path().join("target"))
        .current_dir(dir.path())
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "invalid derives compiled successfully"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    for (name, _, expected) in &cases {
        assert!(
            stderr.contains(expected),
            "{name} stderr did not contain {expected:?}:\n{stderr}"
        );
    }
    assert!(
        !stderr.contains("proc macro panicked"),
        "derive reported a proc-macro panic:\n{stderr}"
    );
}
