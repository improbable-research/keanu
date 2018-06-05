package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary;


import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.NonProbabilisticDoubleTensor;

public abstract class TensorDoubleBinaryOpVertex extends NonProbabilisticDoubleTensor {

    protected final DoubleTensorVertex a;
    protected final DoubleTensorVertex b;

    public TensorDoubleBinaryOpVertex(DoubleTensorVertex a, DoubleTensorVertex b) {
        int[] resultShape = TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), b.getShape());
        this.a = a;
        this.b = b;
        setParents(a, b);
        setValue(DoubleTensor.placeHolder(resultShape));
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op(a.sample(random), b.sample(random));
    }

    @Override
    public DoubleTensor getDerivedValue() {
        return op(a.getValue(), b.getValue());
    }

    protected abstract DoubleTensor op(DoubleTensor a, DoubleTensor b);

}
