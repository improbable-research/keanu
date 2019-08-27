package io.improbable.snippet;

import io.improbable.keanu.vertices.tensor.generic.probabilistic.discrete.CategoricalVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.LinkedHashMap;

import static io.improbable.snippet.SelectVertexExample.MyType.A;
import static io.improbable.snippet.SelectVertexExample.MyType.B;
import static io.improbable.snippet.SelectVertexExample.MyType.C;
import static io.improbable.snippet.SelectVertexExample.MyType.D;

public class SelectVertexExample {

    //%%SNIPPET_START%% VertexSelectVertexCode
    public enum MyType {
        A, B, C, D
    }

    public CategoricalVertex<MyType> getSelectorForMyType() {

        LinkedHashMap<MyType, DoubleVertex> frequency = new LinkedHashMap<>();
        frequency.put(A, new ConstantDoubleVertex(0.25));
        frequency.put(B, new ConstantDoubleVertex(0.25));
        frequency.put(C, new ConstantDoubleVertex(0.25));
        frequency.put(D, new ConstantDoubleVertex(0.25));

        return new CategoricalVertex<>(frequency);
    }
    //%%SNIPPET_END%% VertexSelectVertexCode

}
