from typing import TypeVar, Optional, Iterator

T = TypeVar('T')


class JavaIterator:
    """
    Wraps a python iterator in an object that implements Java's Iterator interface.
    The main difference is that Python has no hasNext() method - it just raises a StopIteration exception when it's finished.
    We implement hasNext() by using a field `next_item` which caches the next item to be returned.
    """

    def __init__(self, python_iterator: Iterator[T]) -> None:
        self.python_iterator = python_iterator
        self.next_item: Optional[T] = None

    def next(self) -> T:
        if self.next_item is None:
            return self.python_iterator.__next__()
        else:
            to_return = self.next_item
            self.next_item = None
            return to_return

    def hasNext(self) -> bool:
        if self.next_item is not None:
            return True

        try:
            self.next_item = self.python_iterator.__next__()
            return True
        except Exception:
            return False

    class Java:
        implements = ["java.util.Iterator"]
