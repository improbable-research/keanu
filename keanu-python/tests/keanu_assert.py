"""
Some utility functions to compare tensor-like things
They might be Keanu tensors or Numpy arrays
"""
def tensors_equal(t1, t2):
    if not shapes_equal(__get_shape(t1), __get_shape(t2)):
        print("Shapes don't match", __get_shape(t1), __get_shape(t2))
        return False
    it1 = __as_list(t1)
    it2 = __as_list(t2)
    for i in range(len(it1)):
        if it1[i] != it2[i]:
            print("Mismatch at position %d: %s != %s" % (i, it1[i], it2[i]))
            return False
    return True

def shapes_equal(s1, s2):
    if __get_length(s1) != __get_length(s2):
        print("Ranks don't match: %s vs %s" % (__as_list(s1), __as_list(s2)))
        return False
    for i in range(len(s2)):
        if s1[i] != s2[i]:
            print("Mismatch in dimension %d: %d != %d" % (i, s1[i], s2[i]))
            return False
        return True

def __get_shape(t):
    try:
        return t.getShape()
    except:
        return t.shape

def __get_length(t):
    try:
        return len(t)
    except:
        return len(__as_list(t))

def __as_list(t):
    try:
        return [i for i in t.asFlatArray()]
    except:
        return [i for i in t]