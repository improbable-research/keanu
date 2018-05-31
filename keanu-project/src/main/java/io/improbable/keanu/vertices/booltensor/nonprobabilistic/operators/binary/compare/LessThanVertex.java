package io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.binary.compare;

import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.binary.BoolBinaryOpVertex;

public class LessThanVertex<TA extends NumberTensor, TB extends NumberTensor> extends BoolBinaryOpVertex<TA, TB> {

    public LessThanVertex(Vertex<TA> a, Vertex<TB> b) {
        super(a, b);
    }

    /**
     * Returns true if a is less than b
     */
    @Override
    public BooleanTensor op(TA a, TB b) {
        return a.toDouble().lessThan(b.toDouble());
    }

}
