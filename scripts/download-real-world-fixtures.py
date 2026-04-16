#!/usr/bin/env python3
"""Download or generate optional real-world HDF5 smoke-test fixtures."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tests" / "data" / "real_world"

DOWNLOADS = [
    (
        "anndataR_example.h5ad",
        "https://raw.githubusercontent.com/scverse/anndataR/devel/inst/extdata/example.h5ad",
    ),
    (
        "keras_conv_mnist_tf_model.h5",
        "https://huggingface.co/osanseviero/keras-conv-mnist/resolve/main/tf_model.h5",
    ),
    (
        "10x_pbmc_1k_v3_filtered_feature_bc_matrix.h5",
        "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_1k_v3/pbmc_1k_v3_filtered_feature_bc_matrix.h5",
    ),
]


def download(name: str, url: str, force: bool) -> None:
    path = OUT / name
    if path.exists() and not force:
        print(f"exists: {path.relative_to(REPO)}")
        return

    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"download: {url}")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "hdf5-pure-rust fixture downloader"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        tmp.write_bytes(response.read())
    tmp.replace(path)
    print(f"wrote: {path.relative_to(REPO)} ({path.stat().st_size} bytes)")


def require_h5py_numpy():
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise SystemExit("h5py and numpy are required for generated real-world fixtures") from exc
    return h5py, np


def generate_h5py_smoke(force: bool) -> None:
    h5py, np = require_h5py_numpy()
    path = OUT / "h5py_3_12_smoke.h5"
    if path.exists() and not force:
        print(f"exists: {path.relative_to(REPO)}")
        return

    with h5py.File(path, "w") as f:
        f.attrs["creator"] = f"h5py {h5py.__version__}"
        f.attrs["unicode"] = "real-world smoke: β 猫"

        run = f.create_group("experiment/run_001")
        run.attrs["temperature_c"] = 21.5
        run.create_dataset(
            "image_stack",
            data=np.arange(24, dtype=np.uint16).reshape(2, 3, 4),
            chunks=(1, 3, 4),
            compression="gzip",
            shuffle=True,
        )
        run.create_dataset(
            "signal",
            data=np.linspace(0.0, 1.0, 25, dtype=np.float64),
            chunks=(10,),
            maxshape=(None,),
        )

        table_dtype = np.dtype([("id", "<i4"), ("score", "<f8")])
        table = np.array([(1, 0.5), (2, 0.75), (3, 1.25)], dtype=table_dtype)
        run.create_dataset("compound_table", data=table)

        string_dtype = h5py.string_dtype(encoding="utf-8")
        run.create_dataset(
            "labels",
            data=np.array(["alpha", "βeta", "猫"], dtype=object),
            dtype=string_dtype,
        )

    print(f"wrote: {path.relative_to(REPO)} ({path.stat().st_size} bytes)")


def generate_netcdf4_like(force: bool) -> None:
    h5py, np = require_h5py_numpy()
    path = OUT / "netcdf4_like_climate.nc"
    if path.exists() and not force:
        print(f"exists: {path.relative_to(REPO)}")
        return

    with h5py.File(path, "w") as f:
        f.attrs["_NCProperties"] = "version=2,hdf5=1.14.5,netcdf=4.9.2"
        f.attrs["Conventions"] = "CF-1.8"
        lat = f.create_dataset("lat", data=np.array([-45.0, 0.0, 45.0], dtype="<f4"))
        lon = f.create_dataset("lon", data=np.array([0.0, 90.0, 180.0, 270.0], dtype="<f4"))
        lat.attrs["units"] = "degrees_north"
        lon.attrs["units"] = "degrees_east"
        lat.make_scale("lat")
        lon.make_scale("lon")

        temp = f.create_dataset(
            "temperature",
            data=np.arange(12, dtype="<f4").reshape(3, 4) + 273.15,
            chunks=(2, 2),
            compression="gzip",
        )
        temp.attrs["units"] = "K"
        temp.attrs["standard_name"] = "air_temperature"
        temp.dims[0].attach_scale(lat)
        temp.dims[1].attach_scale(lon)

    print(f"wrote: {path.relative_to(REPO)} ({path.stat().st_size} bytes)")


def generate_matlab_v73_like(force: bool) -> None:
    h5py, np = require_h5py_numpy()
    path = OUT / "matlab_v73_like.mat"
    if path.exists() and not force:
        print(f"exists: {path.relative_to(REPO)}")
        return

    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")
        refs.attrs["H5PATH"] = "/#refs#"

        a = f.create_dataset("A", data=np.arange(6, dtype="<f8").reshape(2, 3), compression="gzip")
        a.attrs["MATLAB_class"] = "double"

        text = np.array([ord(ch) for ch in "hello"], dtype="<u2").reshape(1, 5)
        name = f.create_dataset("name", data=text)
        name.attrs["MATLAB_class"] = "char"

        value = refs.create_dataset("value", data=np.array([[42.0]], dtype="<f8"))
        value.attrs["MATLAB_class"] = "double"
        ref_dtype = h5py.special_dtype(ref=h5py.Reference)
        cell = f.create_dataset("cell", data=np.array([value.ref], dtype=ref_dtype))
        cell.attrs["MATLAB_class"] = "cell"

    print(f"wrote: {path.relative_to(REPO)} ({path.stat().st_size} bytes)")


def generate_nexus(force: bool) -> None:
    h5py, np = require_h5py_numpy()
    path = OUT / "nexus_simple.nxs"
    if path.exists() and not force:
        print(f"exists: {path.relative_to(REPO)}")
        return

    with h5py.File(path, "w") as f:
        f.attrs["default"] = "entry"
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["default"] = "data"
        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = "NXinstrument"
        detector = instrument.create_group("detector")
        detector.attrs["NX_class"] = "NXdetector"
        detector.create_dataset("counts", data=np.arange(12, dtype="<i4").reshape(3, 4))
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data.attrs["signal"] = "counts"
        data["counts"] = h5py.SoftLink("/entry/instrument/detector/counts")

    print(f"wrote: {path.relative_to(REPO)} ({path.stat().st_size} bytes)")


def generate_pandas_hdfstore(force: bool) -> None:
    path = OUT / "pandas_hdfstore_table.h5"
    if path.exists() and not force:
        print(f"exists: {path.relative_to(REPO)}")
        return

    try:
        import numpy as np
        import pandas as pd
        import tables  # noqa: F401
    except ImportError:
        print("skip: pandas_hdfstore_table.h5 requires pandas with PyTables installed")
        return

    df = pd.DataFrame(
        {
            "sample": ["a", "b", "c", "d"],
            "count": np.array([10, 20, 30, 40], dtype=np.int64),
            "score": np.array([0.5, 0.75, 1.25, 1.5], dtype=np.float64),
        }
    )
    df.to_hdf(path, key="observations", mode="w", format="table", data_columns=["sample"])
    print(f"wrote: {path.relative_to(REPO)} ({path.stat().st_size} bytes)")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite existing fixtures")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="only generate local fixtures; skip public downloads",
    )
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    if not args.no_download:
        for name, url in DOWNLOADS:
            download(name, url, args.force)
    generate_h5py_smoke(args.force)
    generate_netcdf4_like(args.force)
    generate_matlab_v73_like(args.force)
    generate_nexus(args.force)
    generate_pandas_hdfstore(args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
