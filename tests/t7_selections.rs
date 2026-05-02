//! Phase T7: Dataspace and selection tests.

use hdf5_pure_rust::{Dataspace, DataspaceType, File, HyperslabDim, Selection, SelectionType};

const FILE: &str = "tests/data/hdf5_ref/selections_test.h5";
fn open() -> File {
    File::open(FILE).unwrap()
}

// T7a: Dataspace types

#[test]
fn t7a_simple_dataspace() {
    let ds = open().dataset("seq100").unwrap();
    let space = ds.space().unwrap();
    assert!(space.is_simple());
    assert!(!space.is_scalar());
    assert!(!space.is_null());
    assert_eq!(space.ndim(), 1);
    assert_eq!(space.shape(), &[100]);
    assert_eq!(space.size(), 100);
}

#[test]
fn t7a_dataspace_create_copy_and_extent_mutation() {
    let scalar = Dataspace::scalar();
    assert!(scalar.is_scalar());
    assert_eq!(scalar.extent_type(), DataspaceType::Scalar);
    assert!(scalar.has_extent());
    assert_eq!(scalar.npoints_max(), 1);

    let null = Dataspace::null();
    assert!(null.is_null());
    assert_eq!(null.extent_nelem(), 0);

    let mut simple = Dataspace::simple(vec![2, 3], Some(vec![4, u64::MAX])).unwrap();
    assert!(simple.is_simple());
    assert_eq!(
        simple.extent_dims(),
        (&[2, 3][..], Some(&[4, u64::MAX][..]))
    );
    assert_eq!(simple.npoints_max(), u64::MAX);
    assert!(simple.extent_equal(&simple.copy()));
    simple.set_extent_simple(vec![5], None).unwrap();
    assert_eq!(simple.shape(), &[5]);
    assert_eq!(simple.extent_nelem(), 5);
    simple.set_version(1).unwrap();
    assert_eq!(simple.raw_message().version, 1);
    assert!(simple.set_version(3).is_err());
}

#[test]
fn t7a_scalar_dataspace() {
    let ds = open().dataset("scalar_val").unwrap();
    let space = ds.space().unwrap();
    assert!(space.is_scalar());
    assert_eq!(space.ndim(), 0);
    assert_eq!(space.size(), 1);
}

#[test]
fn t7a_null_dataspace() {
    let ds = open().dataset("null_ds").unwrap();
    let space = ds.space().unwrap();
    assert!(space.is_null() || space.is_scalar());
    assert_eq!(space.ndim(), 0);
}

// T7b: Dimension queries

#[test]
fn t7b_2d_shape() {
    let ds = open().dataset("matrix").unwrap();
    let space = ds.space().unwrap();
    assert_eq!(space.shape(), &[6, 10]);
    assert_eq!(space.ndim(), 2);
    assert_eq!(space.size(), 60);
}

#[test]
fn t7b_3d_shape() {
    let ds = open().dataset("cube").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![4, 5, 6]);
}

#[test]
fn t7b_resizable_maxdims() {
    let ds = open().dataset("resizable").unwrap();
    let space = ds.space().unwrap();
    assert!(space.is_resizable());
    let maxdims = space.maxdims().unwrap();
    assert_eq!(maxdims[0], u64::MAX); // unlimited
}

#[test]
fn t7b_non_resizable() {
    let ds = open().dataset("seq100").unwrap();
    assert!(!ds.space().unwrap().is_resizable());
}

// T7c-e: Selection / read_slice tests

#[test]
fn t7c_slice_1d_range() {
    let ds = open().dataset("seq100").unwrap();
    let vals: Vec<f64> = ds.read_slice::<f64, _>(10..20).unwrap();
    assert_eq!(vals.len(), 10);
    for (i, v) in vals.iter().enumerate() {
        assert_eq!(*v, (10 + i) as f64);
    }
}

#[test]
fn t7c_selection_constructor_and_internal_query_aliases() {
    assert_eq!(Selection::select_none(), Selection::None);
    assert_eq!(Selection::select_none_api(), Selection::None);
    assert_eq!(Selection::none_iter_block(), None);
    assert_eq!(Selection::none_iter_nelmts(), 0);
    assert_eq!(Selection::none_iter_get_seq_list(), Vec::<Vec<u64>>::new());
    Selection::none_iter_release();
    Selection::none_release();
    assert_eq!(Selection::none_copy(), Selection::None);
    assert!(Selection::none_is_valid());
    assert_eq!(Selection::none_serialize(), Vec::<u8>::new());
    assert_eq!(Selection::none_deserialize(&[]).unwrap(), Selection::None);
    assert!(Selection::none_deserialize(&[1]).is_err());
    assert_eq!(Selection::none_bounds(), None);
    assert_eq!(Selection::none_offset(&[1, -1]), Selection::None);
    assert!(Selection::none_is_contiguous());
    assert!(!Selection::none_is_single());
    assert!(Selection::none_is_regular());
    assert!(!Selection::none_intersect_block(&[0], &[1]));
    assert_eq!(Selection::none_adjust_u(&[1]), Selection::None);
    assert_eq!(Selection::none_adjust_s(&[-1]), Selection::None);
    assert_eq!(Selection::none_project_simple(), Selection::None);
    assert_eq!(Selection::select_all(), Selection::All);
    assert_eq!(Selection::select_all_api(), Selection::All);
    let mut all_iter = Selection::all_iter_init(&[2, 2]).unwrap();
    assert_eq!(
        Selection::all_iter_coords(&[2, 2]).unwrap(),
        Some(vec![0, 0])
    );
    assert_eq!(
        Selection::all_iter_block(&[2, 2]),
        Some((vec![0, 0], vec![1, 1]))
    );
    assert_eq!(Selection::all_iter_nelmts(&[2, 2]).unwrap(), 4);
    assert!(Selection::all_iter_has_next_block(&[2, 2]));
    assert_eq!(Selection::all_iter_next(&mut all_iter), Some(vec![0, 0]));
    assert_eq!(
        Selection::all_iter_next_block(&[2, 2]),
        Some((vec![0, 0], vec![1, 1]))
    );
    assert_eq!(
        Selection::all_iter_get_seq_list(&[2, 2], 3).unwrap(),
        vec![vec![0, 0], vec![0, 1], vec![1, 0]]
    );
    Selection::all_iter_release(all_iter);
    Selection::all_release();
    assert_eq!(Selection::all_copy(), Selection::All);
    assert!(Selection::all_is_valid(&[2, 2]));
    assert_eq!(Selection::all_serial_size(), 0);
    assert_eq!(Selection::all_serialize(), Vec::<u8>::new());
    assert_eq!(Selection::all_deserialize(&[]).unwrap(), Selection::All);
    assert!(Selection::all_deserialize(&[1]).is_err());
    assert_eq!(
        Selection::all_bounds(&[2, 2]),
        Some((vec![0, 0], vec![1, 1]))
    );
    assert_eq!(Selection::all_offset(&[0, 0]).unwrap(), Selection::All);
    assert!(Selection::all_offset(&[1]).is_err());
    assert_eq!(Selection::all_unlim_dim(&[4, u64::MAX]), Some(1));
    assert!(Selection::all_is_contiguous());
    assert!(Selection::all_is_single(&[]));
    assert!(Selection::all_is_regular());
    assert!(Selection::all_intersect_block(&[3], &[1], &[2]));
    assert!(!Selection::all_intersect_block(&[3], &[3], &[4]));
    assert_eq!(Selection::all_adjust_u(&[0]).unwrap(), Selection::All);
    assert_eq!(Selection::all_adjust_s(&[0]).unwrap(), Selection::All);
    assert_eq!(
        Selection::all_project_simple(&[2, 2], &[1]).unwrap(),
        Selection::Points(vec![vec![0], vec![1]])
    );

    let points = Selection::select_elements(vec![vec![1, 2]]);
    assert_eq!(points.select_npoints_internal(&[3, 4]), Some(1));
    assert_eq!(points.select_type_internal(), SelectionType::Points);
    assert!(points.select_valid(&[3, 4]));
    assert!(!points.select_valid(&[1, 1]));
    assert_eq!(points.select_copy(), points);
    assert!(points.select_is_single(&[3, 4]));
    assert_eq!(
        points.select_bounds_internal(&[3, 4]),
        Some((vec![1, 2], vec![1, 2]))
    );
    assert_eq!(
        Selection::select_elements_api(vec![vec![0]]).element_point_count(),
        Some(1)
    );

    let hyper = Selection::select_hyperslab(vec![HyperslabDim::new(1, 1, 2, 1)]);
    assert_eq!(hyper.selected_count(&[5]), Some(2));
    assert!(hyper.select_is_contiguous(&[5]).unwrap());
    assert!(hyper.select_shape_same(
        &Selection::select_hyperslab(vec![HyperslabDim::new(0, 1, 2, 1)]),
        &[5]
    ));
    assert!(hyper.select_shape_same_api(
        &Selection::select_hyperslab(vec![HyperslabDim::new(0, 1, 2, 1)]),
        &[5]
    ));
    assert_eq!(
        Selection::select_hyperslab_api(vec![HyperslabDim::new(0, 1, 1, 1)]).selection_type(),
        SelectionType::Hyperslab
    );
}

#[test]
fn t7c_selection_iterator_and_adjust_aliases() {
    let selection = Selection::select_hyperslab(vec![
        HyperslabDim::new(0, 2, 2, 1),
        HyperslabDim::new(1, 1, 2, 1),
    ]);

    let mut iter = selection.select_iter_init(&[4, 4]).unwrap();
    assert_eq!(iter.select_iter_nelmts(), 4);
    assert_eq!(iter.select_iter_next(), Some(vec![0, 1]));
    assert_eq!(
        iter.select_iter_get_seq_list(2),
        vec![vec![0, 2], vec![2, 1]]
    );
    assert_eq!(iter.select_iter_nelmts(), 1);
    iter.select_iter_release();

    let mut visited = Vec::new();
    selection
        .select_iterate(&[4, 4], |point| {
            visited.push(point.to_vec());
            Ok(())
        })
        .unwrap();
    assert_eq!(
        visited,
        vec![vec![0, 1], vec![0, 2], vec![2, 1], vec![2, 2]]
    );

    assert_eq!(
        Selection::Points(vec![vec![1, 2]])
            .select_offset(&[2, -1])
            .unwrap(),
        Selection::Points(vec![vec![3, 1]])
    );
    assert!(Selection::Points(vec![vec![0]])
        .select_adjust_signed(&[-1])
        .is_err());
    assert_eq!(
        Selection::Slice(vec![hdf5_pure_rust::SliceInfo::new(1, 3)])
            .select_adjust_unsigned(&[2])
            .unwrap(),
        Selection::Slice(vec![hdf5_pure_rust::SliceInfo::new(3, 5)])
    );
    assert_eq!(
        Selection::select_all().select_adjust_api(&[0, 0]).unwrap(),
        Selection::All
    );
}

#[test]
fn t7c_selection_projection_unlimited_and_fill_aliases() {
    let lhs = Selection::Points(vec![vec![0, 1, 2], vec![3, 1, 4], vec![3, 2, 4]]);
    let rhs = Selection::Points(vec![vec![3, 1, 4], vec![9, 9, 9]]);

    assert_eq!(lhs.select_unlim_dim(&[5, u64::MAX, 6]), Some(1));
    assert_eq!(
        lhs.select_num_elem_non_unlim(&[4, 3, 5], &[4, u64::MAX, 5])
            .unwrap(),
        2
    );
    assert_eq!(
        lhs.select_construct_projection(&[4, 3, 5], &[0, 2])
            .unwrap(),
        Selection::Points(vec![vec![0, 2], vec![3, 4]])
    );
    assert_eq!(
        lhs.select_project_intersection(&rhs, &[4, 3, 5], &[0, 2])
            .unwrap(),
        Selection::Points(vec![vec![3, 4]])
    );
    assert_eq!(
        lhs.select_project_intersection_api(&rhs, &[4, 3, 5], &[1])
            .unwrap(),
        Selection::Points(vec![vec![1]])
    );

    let mut values = vec![0i32; 6];
    Selection::Points(vec![vec![0, 1], vec![1, 2]])
        .select_fill(&[2, 3], &mut values, 7)
        .unwrap();
    assert_eq!(values, vec![0, 7, 0, 0, 0, 7]);
}

#[test]
fn t7c_selection_class_specific_internal_aliases() {
    let hyper = Selection::select_hyperslab(vec![HyperslabDim::new(0, 2, 2, 1)]);
    assert!(hyper.hyper_is_valid(&[4]));
    assert_eq!(hyper.hyper_span_nblocks(&[4]), Some(2));
    assert_eq!(hyper.get_select_hyper_nblocks_internal(&[4]), Some(2));
    assert_eq!(
        hyper.hyper_span_blocklist(&[4]).unwrap(),
        Some(vec![(vec![0], vec![0]), (vec![2], vec![2])])
    );
    assert_eq!(hyper.hyper_bounds(&[4]), Some((vec![0], vec![2])));
    assert!(hyper.hyper_is_regular());
    assert!(hyper.hyper_shape_same(
        &Selection::select_hyperslab(vec![HyperslabDim::new(1, 1, 2, 1)]),
        &[4]
    ));
    assert_eq!(
        hyper.hyper_adjust_u(&[1]).unwrap(),
        Selection::select_hyperslab(vec![HyperslabDim::new(1, 2, 2, 1)])
    );
    assert_eq!(
        hyper.hyper_project_simple(&[4], &[0]).unwrap(),
        Selection::Points(vec![vec![0], vec![2]])
    );

    let points = Selection::Points(vec![vec![0, 1], vec![1, 1]]);
    assert_eq!(
        Selection::copy_pnt_list(points.element_pointlist().unwrap()),
        vec![vec![0, 1], vec![1, 1]]
    );
    Selection::free_pnt_list(vec![vec![0, 1]]);
    let mut grown_points = points.clone();
    grown_points.point_add(vec![1, 0]).unwrap();
    assert_eq!(
        grown_points,
        Selection::Points(vec![vec![0, 1], vec![1, 1], vec![1, 0]])
    );
    assert!(grown_points.point_add(vec![0, 0, 0]).is_err());
    let mut iter = points.point_iter_init(&[2, 2]).unwrap();
    assert_eq!(iter.point_iter_coords(), Some(&[0, 1][..]));
    assert_eq!(iter.point_iter_nelmts(), 2);
    assert_eq!(iter.point_iter_next_block(), Some(vec![0, 1]));
    assert_eq!(iter.point_iter_get_seq_list(2), vec![vec![1, 1]]);
    assert_eq!(
        points.get_select_elem_pointlist_internal().unwrap(),
        &[vec![0, 1], vec![1, 1]][..]
    );
    assert!(points.point_is_valid(&[2, 2]));
    assert!(!points.point_is_regular());
    assert!(points.point_shape_same(&Selection::Points(vec![vec![9, 9], vec![8, 8]]), &[2, 2]));
    assert!(points.point_intersect_block(&[0, 0], &[0, 1]));
    assert!(!points.point_intersect_block(&[2, 0], &[3, 1]));
    assert_eq!(
        points.point_adjust_s(&[1, -1]).unwrap(),
        Selection::Points(vec![vec![1, 0], vec![2, 0]])
    );
}

#[test]
fn t7c_slice_1d_from_start() {
    let ds = open().dataset("seq100").unwrap();
    let vals: Vec<f64> = ds.read_slice::<f64, _>(..5).unwrap();
    assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn t7c_slice_1d_to_end() {
    let ds = open().dataset("seq100").unwrap();
    let vals: Vec<f64> = ds.read_slice::<f64, _>(95..).unwrap();
    assert_eq!(vals, vec![95.0, 96.0, 97.0, 98.0, 99.0]);
}

#[test]
fn t7c_slice_all() {
    let ds = open().dataset("seq100").unwrap();
    let vals: Vec<f64> = ds.read_slice::<f64, _>(..).unwrap();
    assert_eq!(vals.len(), 100);
}

#[test]
fn t7d_slice_2d() {
    let ds = open().dataset("matrix").unwrap();
    // Read rows 1-3, cols 2-5
    let vals: Vec<i32> = ds.read_slice::<i32, _>((1..3, 2..5)).unwrap();
    // row 1: [10,11,12,13,14,15,16,17,18,19], cols 2-4 = [12,13,14]
    // row 2: [20,21,22,23,24,25,26,27,28,29], cols 2-4 = [22,23,24]
    assert_eq!(vals, vec![12, 13, 14, 22, 23, 24]);
}

#[test]
fn t7d_slice_2d_single_row() {
    let ds = open().dataset("matrix").unwrap();
    let vals: Vec<i32> = ds.read_slice::<i32, _>((0..1, 0..10)).unwrap();
    assert_eq!(vals, (0..10).collect::<Vec<i32>>());
}

#[test]
fn t7d_slice_3d_subregion() {
    let ds = open().dataset("cube").unwrap();
    let vals: Vec<i32> = ds.read_slice::<i32, _>((1..3, 2..4, 1..4)).unwrap();
    assert_eq!(vals, vec![43, 44, 45, 49, 50, 51, 73, 74, 75, 79, 80, 81]);
}

#[test]
fn t7d_slice_3d_with_full_dimension() {
    let ds = open().dataset("cube").unwrap();
    let vals: Vec<i32> = ds.read_slice::<i32, _>((2..3, .., 4..)).unwrap();
    assert_eq!(vals, vec![64, 65, 70, 71, 76, 77, 82, 83, 88, 89]);
}

#[test]
fn t7d_point_selection_3d() {
    let ds = open().dataset("cube").unwrap();
    let vals: Vec<i32> = ds
        .read_slice::<i32, _>(Selection::Points(vec![
            vec![3, 4, 5],
            vec![0, 0, 0],
            vec![1, 2, 3],
        ]))
        .unwrap();
    assert_eq!(vals, vec![119, 0, 45]);
}

#[test]
fn t7e_slice_chunked() {
    let ds = open().dataset("chunked_seq").unwrap();
    // Slice across chunk boundary (chunks are 25 elements)
    let vals: Vec<i64> = ds.read_slice::<i64, _>(20..30).unwrap();
    assert_eq!(vals.len(), 10);
    for (i, v) in vals.iter().enumerate() {
        assert_eq!(*v, (20 + i) as i64);
    }
}

#[test]
fn t7e_slice_chunked_all() {
    let ds = open().dataset("chunked_seq").unwrap();
    let vals: Vec<i64> = ds.read_slice::<i64, _>(..).unwrap();
    assert_eq!(vals.len(), 200);
    assert_eq!(vals[0], 0);
    assert_eq!(vals[199], 199);
}

#[test]
fn t7e_read_scalar() {
    let ds = open().dataset("scalar_val").unwrap();
    let val: f64 = ds.read_scalar::<f64>().unwrap();
    assert_eq!(val, 42.0);
}
