from typing import Tuple

def check_index_is_valid(shape: Tuple[int], index: Tuple[int]):
    if len(shape) != len(index):
        raise ValueError("Length of desired index {} must match the length of the shape {}.".format(index, shape))
    for i in range(len(index)):
      if index[i] >= shape[i]:
          raise ValueError("Invalid index {} for shape {}".format(index, shape))