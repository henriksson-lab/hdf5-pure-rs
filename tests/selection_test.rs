use hdf5_pure_rust::{
    hl::selection::{Selection, SliceInfo},
    File, HyperslabDim, SelectionType,
};

#[test]
fn test_read_slice_1d_range() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    // Read elements 1..4 (indices 1, 2, 3)
    let vals: Vec<f64> = ds.read_slice::<f64, _>(1..4).unwrap();
    assert_eq!(vals, vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_read_slice_1d_from_start() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let vals: Vec<f64> = ds.read_slice::<f64, _>(..3).unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_read_slice_1d_to_end() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let vals: Vec<f64> = ds.read_slice::<f64, _>(3..).unwrap();
    assert_eq!(vals, vec![4.0, 5.0]);
}

#[test]
fn test_read_slice_all() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let vals: Vec<f64> = ds.read_slice::<f64, _>(..).unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_read_slice_none() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let vals: Vec<f64> = ds.read_slice::<f64, _>(Selection::None).unwrap();
    assert!(vals.is_empty());
}

#[test]
fn test_read_slice_zero_length_range() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let vals: Vec<f64> = ds.read_slice::<f64, _>(2..2).unwrap();
    assert!(vals.is_empty());
}

#[test]
fn test_read_slice_1d_stepped_selection() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let selection = Selection::Slice(vec![SliceInfo::with_step(0, 5, 2)]);
    let vals: Vec<f64> = ds.read_slice::<f64, _>(selection).unwrap();
    assert_eq!(vals, vec![1.0, 3.0, 5.0]);
}

#[test]
fn test_read_slice_1d_point_selection() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let selection = Selection::Points(vec![vec![4], vec![0], vec![2]]);
    let vals: Vec<f64> = ds.read_slice::<f64, _>(selection).unwrap();
    assert_eq!(vals, vec![5.0, 1.0, 3.0]);
}

#[test]
fn test_read_slice_1d_block_hyperslab_selection() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let selection = Selection::Hyperslab(vec![HyperslabDim::new(0, 3, 2, 2)]);
    let vals: Vec<f64> = ds.read_slice::<f64, _>(selection).unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 4.0, 5.0]);
}

#[test]
fn test_read_slice_2d() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    // Read row 1 only (row 1, all columns)
    let vals: Vec<i8> = ds.read_slice::<i8, _>((1..2, 0..3)).unwrap();
    assert_eq!(vals, vec![4, 5, 6]);
}

#[test]
fn test_read_slice_2d_range_full_column() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    let vals: Vec<i8> = ds.read_slice::<i8, _>((.., 1..3)).unwrap();
    assert_eq!(vals, vec![2, 3, 5, 6]);
}

#[test]
fn test_read_slice_2d_range_full_row() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    let vals: Vec<i8> = ds.read_slice::<i8, _>((1..2, ..)).unwrap();
    assert_eq!(vals, vec![4, 5, 6]);
}

#[test]
fn test_read_slice_2d_full_tuple() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    let vals: Vec<i8> = ds.read_slice::<i8, _>((.., ..)).unwrap();
    assert_eq!(vals, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_read_slice_2d_open_ended_tuple_ranges() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    let vals: Vec<i8> = ds.read_slice::<i8, _>((1.., ..2)).unwrap();
    assert_eq!(vals, vec![4, 5]);
}

#[test]
fn test_read_slice_2d_subregion() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    // Read rows 0-1, cols 1-2
    let vals: Vec<i8> = ds.read_slice::<i8, _>((0..2, 1..3)).unwrap();
    assert_eq!(vals, vec![2, 3, 5, 6]);
}

#[test]
fn test_read_slice_2d_stepped_selection() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    let selection = Selection::Slice(vec![
        SliceInfo::with_step(0, 2, 1),
        SliceInfo::with_step(0, 3, 2),
    ]);
    let vals: Vec<i8> = ds.read_slice::<i8, _>(selection).unwrap();
    assert_eq!(vals, vec![1, 3, 4, 6]);
}

#[test]
fn test_read_slice_2d_point_selection() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    let selection = Selection::Points(vec![vec![1, 2], vec![0, 0], vec![1, 1]]);
    let vals: Vec<i8> = ds.read_slice::<i8, _>(selection).unwrap();
    assert_eq!(vals, vec![6, 1, 5]);
}

#[test]
fn test_read_slice_2d_block_hyperslab_selection() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    let selection = Selection::Hyperslab(vec![
        HyperslabDim::new(0, 1, 2, 1),
        HyperslabDim::new(0, 2, 2, 1),
    ]);
    let vals: Vec<i8> = ds.read_slice::<i8, _>(selection).unwrap();
    assert_eq!(vals, vec![1, 3, 4, 6]);
}

#[test]
fn test_selection_bounds_count_and_regularity_helpers() {
    let shape = [2, 3];

    let all = Selection::All;
    assert!(all.is_all());
    assert_eq!(all.selection_type(), SelectionType::All);
    assert_eq!(all.selected_count(&shape), Some(6));
    assert_eq!(all.bounds(&shape), Some((vec![0, 0], vec![1, 2])));
    assert_eq!(all.hyperslab_block_count(&shape), Some(1));
    assert_eq!(
        all.hyperslab_blocklist(&shape).unwrap(),
        Some(vec![(vec![0, 0], vec![1, 2])])
    );
    assert!(all.is_regular());

    let none = Selection::None;
    assert!(none.is_none());
    assert_eq!(none.selection_type(), SelectionType::None);
    assert_eq!(none.selected_count(&shape), Some(0));
    assert_eq!(none.bounds(&shape), None);
    assert_eq!(none.hyperslab_block_count(&shape), Some(0));
    assert_eq!(none.hyperslab_blocklist(&shape).unwrap(), Some(Vec::new()));
    assert!(none.is_regular());

    let points = Selection::Points(vec![vec![1, 2], vec![0, 1], vec![1, 0]]);
    assert_eq!(points.selection_type(), SelectionType::Points);
    assert_eq!(points.selected_count(&shape), Some(3));
    assert_eq!(points.bounds(&shape), Some((vec![0, 0], vec![1, 2])));
    assert_eq!(points.hyperslab_block_count(&shape), None);
    assert_eq!(points.hyperslab_blocklist(&shape).unwrap(), None);
    assert_eq!(points.element_point_count(), Some(3));
    assert_eq!(
        points.element_pointlist(),
        Some(vec![vec![1, 2], vec![0, 1], vec![1, 0]].as_slice())
    );
    assert!(!points.is_regular());

    let slices = Selection::Slice(vec![
        SliceInfo::with_step(0, 2, 1),
        SliceInfo::with_step(0, 3, 2),
    ]);
    assert_eq!(slices.selection_type(), SelectionType::Hyperslab);
    assert_eq!(slices.selected_count(&shape), Some(4));
    assert_eq!(slices.bounds(&shape), Some((vec![0, 0], vec![1, 2])));
    assert_eq!(slices.hyperslab_block_count(&shape), Some(2));
    assert_eq!(
        slices.hyperslab_blocklist(&shape).unwrap(),
        Some(vec![(vec![0, 0], vec![1, 0]), (vec![0, 2], vec![1, 2])])
    );
    assert_eq!(slices.element_point_count(), None);
    assert!(slices.is_regular());

    let hyperslab = Selection::Hyperslab(vec![
        HyperslabDim::new(0, 1, 2, 1),
        HyperslabDim::new(0, 2, 2, 1),
    ]);
    assert_eq!(hyperslab.selection_type(), SelectionType::Hyperslab);
    assert_eq!(hyperslab.selected_count(&shape), Some(4));
    assert_eq!(hyperslab.bounds(&shape), Some((vec![0, 0], vec![1, 2])));
    assert_eq!(hyperslab.hyperslab_block_count(&shape), Some(4));
    assert_eq!(
        hyperslab.hyperslab_blocklist(&shape).unwrap(),
        Some(vec![
            (vec![0, 0], vec![0, 0]),
            (vec![0, 2], vec![0, 2]),
            (vec![1, 0], vec![1, 0]),
            (vec![1, 2], vec![1, 2])
        ])
    );
    assert!(hyperslab.is_regular());
}

#[test]
fn test_selection_materialize_and_combine_helpers() {
    let shape = [2, 3];
    let left = Selection::Slice(vec![SliceInfo::new(0, 2), SliceInfo::with_step(0, 3, 2)]);
    let right = Selection::Points(vec![vec![0, 1], vec![1, 2]]);

    assert_eq!(
        left.materialize_points(&shape).unwrap(),
        vec![vec![0, 0], vec![0, 2], vec![1, 0], vec![1, 2]]
    );

    assert_eq!(
        left.combine_or(&right, &shape).unwrap(),
        Selection::Points(vec![
            vec![0, 0],
            vec![0, 1],
            vec![0, 2],
            vec![1, 0],
            vec![1, 2],
        ])
    );
    assert_eq!(
        left.combine_and(&right, &shape).unwrap(),
        Selection::Points(vec![vec![1, 2]])
    );
    assert_eq!(
        left.combine_xor(&right, &shape).unwrap(),
        Selection::Points(vec![vec![0, 0], vec![0, 1], vec![0, 2], vec![1, 0]])
    );
    assert_eq!(
        left.combine_and_not(&right, &shape).unwrap(),
        Selection::Points(vec![vec![0, 0], vec![0, 2], vec![1, 0]])
    );
}

#[test]
fn test_selection_linear_bounds_and_contiguity_helpers() {
    let shape = [3, 4];

    let all = Selection::All;
    assert_eq!(all.linear_bounds(&shape).unwrap(), Some((0, 11)));
    assert!(all.is_contiguous(&shape).unwrap());

    let none = Selection::None;
    assert_eq!(none.linear_bounds(&shape).unwrap(), None);
    assert!(none.is_contiguous(&shape).unwrap());

    let row = Selection::Slice(vec![SliceInfo::new(1, 2), SliceInfo::new(0, 4)]);
    assert_eq!(row.linear_bounds(&shape).unwrap(), Some((4, 7)));
    assert!(row.is_contiguous(&shape).unwrap());

    let column = Selection::Slice(vec![SliceInfo::new(0, 3), SliceInfo::new(1, 2)]);
    assert_eq!(column.linear_bounds(&shape).unwrap(), Some((1, 9)));
    assert!(!column.is_contiguous(&shape).unwrap());

    let explicit = Selection::Points(vec![vec![0, 2], vec![0, 1], vec![0, 3]]);
    assert_eq!(explicit.linear_bounds(&shape).unwrap(), Some((1, 3)));
    assert!(explicit.is_contiguous(&shape).unwrap());

    let duplicate = Selection::Points(vec![vec![0, 1], vec![0, 1]]);
    assert!(!duplicate.is_contiguous(&shape).unwrap());
}

#[test]
fn test_selection_point_iterator_and_projection_helpers() {
    let shape = [2, 3];
    let selection = Selection::Slice(vec![SliceInfo::new(0, 2), SliceInfo::with_step(0, 3, 2)]);

    let mut iter = selection.iter_points(&shape).unwrap();
    assert_eq!(iter.len(), 4);
    assert_eq!(iter.next(), Some(vec![0, 0]));
    assert_eq!(iter.next(), Some(vec![0, 2]));
    assert_eq!(iter.len(), 2);
    assert_eq!(iter.collect::<Vec<_>>(), vec![vec![1, 0], vec![1, 2]]);

    assert_eq!(
        selection.project(&shape, &[0]).unwrap(),
        Selection::Points(vec![vec![0], vec![1]])
    );
    assert_eq!(
        selection.project(&shape, &[1]).unwrap(),
        Selection::Points(vec![vec![0], vec![2]])
    );
    assert_eq!(
        selection.project(&shape, &[1, 0]).unwrap(),
        Selection::Points(vec![vec![0, 0], vec![0, 1], vec![2, 0], vec![2, 1]])
    );

    let err = selection
        .project(&shape, &[2])
        .expect_err("projection should reject out-of-rank dimensions");
    assert!(err.to_string().contains("out of bounds"));
}

#[test]
fn test_selection_bounds_reject_coordinate_overflow() {
    let slice = Selection::Slice(vec![SliceInfo::with_step(u64::MAX - 1, u64::MAX, 2)]);
    assert_eq!(
        slice.bounds(&[u64::MAX]),
        Some((vec![u64::MAX - 1], vec![u64::MAX - 1]))
    );

    let zero_step_slice = Selection::Slice(vec![SliceInfo::with_step(1, u64::MAX, 0)]);
    assert_eq!(zero_step_slice.bounds(&[u64::MAX]), None);

    let overflowing_hyperslab =
        Selection::Hyperslab(vec![HyperslabDim::new(u64::MAX - 1, 2, 2, 1)]);
    assert_eq!(overflowing_hyperslab.bounds(&[u64::MAX]), None);
}

#[test]
fn test_read_slice_chunked() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("chunked").unwrap();
    // Read elements 50..55 from a chunked dataset
    let vals: Vec<f32> = ds.read_slice::<f32, _>(50..55).unwrap();
    assert_eq!(vals, vec![50.0, 51.0, 52.0, 53.0, 54.0]);
}
