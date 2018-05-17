package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

import java.util.Map;

import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticVertexShaping.checkParentShapes;

public class TensorExponentialVertex extends ProbabilisticDoubleTensor {

    private final DoubleTensorVertex a;
    private final DoubleTensorVertex b;
    private final KeanuRandom random;

    public TensorExponentialVertex(int[] shape, DoubleTensorVertex a, DoubleTensorVertex b, KeanuRandom random) {

        checkParentShapes(shape, a.getValue(), b.getValue());

        this.a = a;
        this.b = b;
        this.random = random;
        setParents(a, b);
        setValue(DoubleTensor.placeHolder(shape));
    }

    @Override
    public double logPdf(DoubleTensor value) {
        return 0;
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        return null;
    }

    @Override
    public DoubleTensor sample() {
        return null;
    }
}


