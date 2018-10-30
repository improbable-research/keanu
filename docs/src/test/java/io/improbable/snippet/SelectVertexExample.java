package io.improbable.snippet;

import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.generic.probabilistic.discrete.CategoricalVertex;

import java.util.LinkedHashMap;

import static io.improbable.snippet.SelectVertexExample.MyType.*;

public class SelectVertexExample {

//%%SNIPPET_START%% VertexSelectVertexCode
public enum MyType {
    A, B, C, D
}

public CategoricalVertex<MyType, GenericTensor<MyType>> getSelectorForMyType() {

    LinkedHashMap<MyType, DoubleVertex> frequency = new LinkedHashMap<>();
    frequency.put(A, new ConstantDoubleVertex(0.25));
    frequency.put(B, new ConstantDoubleVertex(0.25));
    frequency.put(C, new ConstantDoubleVertex(0.25));
    frequency.put(D, new ConstantDoubleVertex(0.25));

    return new CategoricalVertex<>(frequency);
}
//%%SNIPPET_END%% VertexSelectVertexCode

}
