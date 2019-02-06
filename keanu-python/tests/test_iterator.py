import pytest

from keanu.functional import JavaIterator


def test_it_tells_you_when_it_is_finished() -> None:
    python_iterator = (i for i in range(5))
    java_iterator = JavaIterator(python_iterator)

    for i in range(5):
        assert(java_iterator.hasNext())
        assert(java_iterator.next() == i)

    assert not java_iterator.hasNext()

def test_it_throws_if_you_pass_the_end() -> None:
    python_iterator = (i for i in range(5))
    java_iterator = JavaIterator(python_iterator)

    for i in range(5):
        java_iterator.next()

    with pytest.raises(StopIteration):
        java_iterator.next()


def test_an_empty_python_iterator_yields_an_empty_java_iterator() -> None:
    python_iterator = (i for i in range(0))
    java_iterator = JavaIterator(python_iterator)

    assert not java_iterator.hasNext()

    with pytest.raises(StopIteration):
        java_iterator.next()
