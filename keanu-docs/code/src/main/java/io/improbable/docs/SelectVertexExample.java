package io.improbable.docs;

import static io.improbable.docs.SelectVertexExample.MyType.A;
import static io.improbable.docs.SelectVertexExample.MyType.B;
import static io.improbable.docs.SelectVertexExample.MyType.C;
import static io.improbable.docs.SelectVertexExample.MyType.D;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.generic.probabilistic.discrete.CategoricalVertex;
import java.util.LinkedHashMap;

public class SelectVertexExample {

  public enum MyType {
    A,
    B,
    C,
    D
  }

  public CategoricalVertex<MyType> getSelectorForMyType() {

    LinkedHashMap<MyType, DoubleVertex> frequency = new LinkedHashMap<>();
    frequency.put(A, ConstantVertex.of(0.25));
    frequency.put(B, ConstantVertex.of(0.25));
    frequency.put(C, ConstantVertex.of(0.25));
    frequency.put(D, ConstantVertex.of(0.25));

    return new CategoricalVertex<MyType>(frequency);
  }
}
