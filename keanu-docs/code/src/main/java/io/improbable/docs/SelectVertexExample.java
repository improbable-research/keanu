package io.improbable.docs;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.generic.probabilistic.discrete.SelectVertex;

import java.util.LinkedHashMap;

import static io.improbable.docs.SelectVertexExample.MyType.*;

public class SelectVertexExample {

    public enum MyType {
        A, B, C, D
    }

    public SelectVertex<MyType> getSelectorForMyType() {

        LinkedHashMap<MyType, DoubleVertex> frequency = new LinkedHashMap<>();
        frequency.put(A, ConstantVertex.of(0.25));
        frequency.put(B, ConstantVertex.of(0.25));
        frequency.put(C, ConstantVertex.of(0.25));
        frequency.put(D, ConstantVertex.of(0.25));

        return new SelectVertex<MyType>(frequency);
    }

}
