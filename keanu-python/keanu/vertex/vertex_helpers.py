from .base import Double, Integer, Boolean, Vertex

def do_vertex_cast(vertex_ctor, value):
    return value if isinstance(value, Vertex) else vertex_ctor(value)