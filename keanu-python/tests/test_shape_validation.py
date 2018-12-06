from keanu.shape_validation import check_index_is_valid
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