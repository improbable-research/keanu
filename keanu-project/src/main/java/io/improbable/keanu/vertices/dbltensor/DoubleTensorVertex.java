package io.improbable.keanu.vertices.dbltensor;


import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary.DifferenceVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary.MultiplicationVertex;

public abstract class DoubleTensorVertex extends TensorVertex<DoubleTensor> {

    public abstract DualNumber getDualNumber();

    public DoubleTensorVertex minus(DoubleTensorVertex that) {
        return new DifferenceVertex(this, that);
    }

    public DoubleTensorVertex plus(DoubleTensorVertex that) {
        return new AdditionVertex(this, that);
    }

    public DoubleTensorVertex multiply(DoubleTensorVertex that) {
        return new MultiplicationVertex(this, that);
    }

}
