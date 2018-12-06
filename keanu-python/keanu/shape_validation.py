from typing import Tuple, List, Union


def check_index_is_valid(shape: Tuple[int, ...], index: Union[Tuple[int], Tuple]) -> None:
    if len(shape) != len(index):
        raise ValueError("Length of desired index {} must match the length of the shape {}.".format(index, shape))
    for i in range(len(index)):
        if index[i] >= shape[i]:
            raise ValueError("Invalid index {} for shape {}".format(index, shape))


def check_all_shapes_match(shapes: List[Tuple[int, ...]]) -> None:
    if not len(set(shapes)) == 1:
        raise ValueError("Shapes must match")
