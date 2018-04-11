package io.improbable.keanu.vertices.dbl.tensor;


import io.improbable.keanu.vertices.dbl.tensor.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.tensor.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.tensor.nonprobabilistic.operators.binary.DifferenceVertex;
import io.improbable.keanu.vertices.dbl.tensor.nonprobabilistic.operators.binary.MultiplicationVertex;

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
