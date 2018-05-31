package io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators;

import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.booltensor.nonprobabilistic.NonProbabilisticBool;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

/**
 * Returns true if a vertex value is equal to another vertex value within an epsilon.
 */
public class NumericalEqualsVertex extends NonProbabilisticBool {

    protected Vertex<? extends NumberTensor> a;
    protected Vertex<? extends NumberTensor> b;
    private Vertex<? extends NumberTensor> epsilon;

    public NumericalEqualsVertex(Vertex<? extends NumberTensor> a,
                                 Vertex<? extends NumberTensor> b,
                                 Vertex<? extends NumberTensor> epsilon) {
        int[] resultShape = TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar(
            a.getShape(),
            b.getShape(),
            epsilon.getShape()
        );
        this.a = a;
        this.b = b;
        this.epsilon = epsilon;
        setParents(a, b, epsilon);
        setValue(BooleanTensor.placeHolder(resultShape));
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return op(a.sample(random), b.sample(random), epsilon.sample(random));
    }

    @Override
    public BooleanTensor getDerivedValue() {
        return op(a.getValue(), b.getValue(), epsilon.getValue());
    }

    private BooleanTensor op(NumberTensor a, NumberTensor b, NumberTensor epsilon) {
        DoubleTensor absoluteDifference = a.toDouble().minus(b.toDouble()).absInPlace();
        return absoluteDifference.lessThanOrEqual(epsilon.toDouble());
    }

}
