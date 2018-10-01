from py4j.java_gateway import java_import
from keanu.base import Vertex, KeanuContext

k = KeanuContext().jvm_view()


java_import(k, "io.improbable.keanu.plating.Plate")
java_import(k, "io.improbable.keanu.python.Keanu")


class Model:
    def __init__(self, vertices={}):
        self.__dict__["_vertices"] = {}
        self.__dict__["_vertices"].update(vertices)

    def __setattr__(self, k, v):
        if k in self.__dict__:
            super(Model, self).__setattr__(k, v)
        else:
            self._vertices[k] = v

    def __getattr__(self, k):
        if k in self.__dict__:
            return self.__dict__[k]
        return self._vertices[k]

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def to_plate(self):
        plate = k.Plate()
        for key, vertex in self._vertices.items():
            plate.add(vertex.unwrap())

        return plate

    @staticmethod
    def from_java(java_model):
        return Model({k: Vertex(None, instance=v) for k, v in java_model.getVertices().items()})
