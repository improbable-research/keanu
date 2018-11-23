from typing import Any
import keanu as kn


class VertexOps:
    """
    __array_ufunc__ is a NumPy thing that enables you to intercept and handle the numpy operation.
    Without this the right operators would fail.
    See https://docs.scipy.org/doc/numpy-1.13.0/neps/ufunc-overrides.html
    """

    def __array_ufunc__(self, ufunc: Any, method: Any, *inputs: Any, **kwargs: Any) -> Any:
        methods = {
            "equal": VertexOps.__eq__,
            "not_equal": VertexOps.__ne__,
            "add": VertexOps.__radd__,
            "subtract": VertexOps.__rsub__,
            "multiply": VertexOps.__rmul__,
            "power": VertexOps.__rpow__,
            "true_divide": VertexOps.__rtruediv__,
            "floor_divide": VertexOps.__rfloordiv__,
            "greater": VertexOps.__lt__,
            "greater_equal": VertexOps.__le__,
            "less": VertexOps.__gt__,
            "less_equal": VertexOps.__ge__,
        }
        if method == "__call__":
            try:
                dispatch_method = methods[ufunc.__name__]
                return dispatch_method(inputs[1], inputs[0])
            except KeyError:
                raise NotImplementedError("NumPy ufunc of type %s not implemented" % ufunc.__name__)
        else:
            raise NotImplementedError("NumPy ufunc method %s not implemented" % method)

    def __add__(self, other: Any) -> Any:
        return kn.vertex.generated.Addition(self, other)

    def __radd__(self, other: Any) -> Any:
        return kn.vertex.generated.Addition(other, self)

    def __sub__(self, other: Any) -> Any:
        return kn.vertex.generated.Difference(self, other)

    def __rsub__(self, other: Any) -> Any:
        return kn.vertex.generated.Difference(other, self)

    def __mul__(self, other: Any) -> Any:
        return kn.vertex.generated.Multiplication(self, other)

    def __rmul__(self, other: Any) -> Any:
        return kn.vertex.generated.Multiplication(other, self)

    def __pow__(self, other: Any) -> Any:
        return kn.vertex.generated.Power(self, other)

    def __rpow__(self, other: Any) -> Any:
        return kn.vertex.generated.Power(other, self)

    def __truediv__(self, other: Any) -> Any:
        return kn.vertex.generated.Division(self, other)

    def __rtruediv__(self, other: Any) -> Any:
        return kn.vertex.generated.Division(other, self)

    def __floordiv__(self, other: Any) -> Any:
        return kn.vertex.generated.IntegerDivision(self, other)

    def __rfloordiv__(self, other: Any) -> Any:
        return kn.vertex.generated.IntegerDivision(other, self)

    def __eq__(self, other: Any) -> Any:
        return kn.vertex.generated.Equals(self, other)

    def __req__(self, other: Any) -> Any:
        return kn.vertex.generated.Equals(self, other)

    def __ne__(self, other: Any) -> Any:
        return kn.vertex.generated.NotEquals(self, other)

    def __rne__(self, other: Any) -> Any:
        return kn.vertex.generated.NotEquals(self, other)

    def __gt__(self, other: Any) -> Any:
        return kn.vertex.generated.GreaterThan(self, other)

    def __ge__(self, other: Any) -> Any:
        return kn.vertex.generated.GreaterThanOrEqual(self, other)

    def __lt__(self, other: Any) -> Any:
        return kn.vertex.generated.LessThan(self, other)

    def __le__(self, other: Any) -> Any:
        return kn.vertex.generated.LessThanOrEqual(self, other)

    def __abs__(self) -> Any:
        return kn.vertex.generated.Abs(self)

    def __round__(self) -> Any:
        return kn.vertex.generated.Round(self)

    def __floor__(self) -> Any:
        return kn.vertex.generated.Floor(self)

    def __ceil__(self) -> Any:
        return kn.vertex.generated.Ceil(self)
