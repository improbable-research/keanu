from keanu.shape_validation import check_index_is_valid, check_all_shapes_match
import pytest

def test_doesnt_throw_on_valid_index():
    check_index_is_valid((3,3),(2,2))

def test_throws_on_index_longer_than_shape():
    with pytest.raises(ValueError):
        check_index_is_valid((2,2),(3,1,3))

def test_throws_on_index_shorter_than_shape():
    with pytest.raises(ValueError):
        check_index_is_valid((2,2),(3,))

def test_throws_on_index_out_of_bounds():
    with pytest.raises(ValueError):
        check_index_is_valid((2, 2,6), (1,1,8))

def test_doesnt_throw_on_matching_shapes():
    shape = (3,3)
    shape_list = [shape,shape,shape]
    check_all_shapes_match(shape_list)

def test_throws_on_non_mathcing_shapes():
    shape_a = (3,4)
    shape_b = (1,3)
    shape_list = [shape_a,shape_a,shape_b]
    with pytest.raises(ValueError):
        check_all_shapes_match(shape_list)

def test_throws_on_empty_list_of_shapes():
    with pytest.raises(ValueError):
        check_all_shapes_match([])