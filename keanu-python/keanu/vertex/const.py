from .generated import ConstantBool, ConstantInteger, ConstantDouble
from .base import Vertex
from keanu.vartypes import tensor_arg_types
from keanu.tensor import cast_double, cast_integer, cast_bool

class Double(Vertex):
    @staticmethod
    def const(arg: tensor_arg_types) -> ConstantDouble:
        return ConstantDouble(cast_double(arg))

    def observe(self, v: tensor_arg_types) -> None:
        self.unwrap().observe(cast_double(v))

    def set_value(self, v: tensor_arg_types) -> None:
        self.unwrap().setValue(cast_double(v))

    def set_and_cascade(self, v: tensor_arg_types) -> None:
        self.unwrap().setAndCascade(cast_double(v))


class Integer(Vertex):
    @staticmethod
    def const(arg: tensor_arg_types) -> ConstantInteger:
        return ConstantInteger(cast_integer(arg))

    def observe(self, v: tensor_arg_types) -> None:
        self.unwrap().observe(cast_integer(arg))

    def set_value(self, v: tensor_arg_types) -> None:
        self.unwrap().setValue(cast_integer(arg))

    def set_and_cascade(self, v: tensor_arg_types) -> None:
        self.unwrap().setAndCascade(cast_integer(arg))


class Bool(Vertex):
    @staticmethod
    def const(arg: tensor_arg_types) -> ConstantBool:
        return ConstantBool(cast_bool(arg))

    def observe(self, v: tensor_arg_types) -> None:
        self.unwrap().observe(cast_bool(arg))

    def set_value(self, v: tensor_arg_types) -> None:
        self.unwrap().setValue(cast_bool(arg))

    def set_and_cascade(self, v: tensor_arg_types) -> None:
        self.unwrap().setAndCascade(cast_bool(arg))
