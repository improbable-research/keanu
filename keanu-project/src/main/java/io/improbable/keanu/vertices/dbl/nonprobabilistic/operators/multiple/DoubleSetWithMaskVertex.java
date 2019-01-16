package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;

public class DoubleSetWithMaskVertex extends DoubleVertex implements NonProbabilistic<DoubleTensor> {

    private final DoubleVertex inputVertex;
    private final DoubleVertex maskVertex;
    private final double value;

    public DoubleSetWithMaskVertex(DoubleVertex inputVertex, DoubleVertex maskVertex, double value) {
        super(checkHasOneNonLengthOneShapeOrAllLengthOne(inputVertex.getShape(), maskVertex.getShape()));
        this.inputVertex = inputVertex;
        this.maskVertex = maskVertex;
        this.value = value;
        setParents(inputVertex, maskVertex);
    }

    @Override
    public DoubleTensor calculate() {
        return inputVertex.getValue().setWithMask(maskVertex.getValue(), value);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return this.getValue();
    }
}
