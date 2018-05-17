package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

import java.util.Map;

import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticVertexShaping.checkParentShapes;
import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticVertexShaping.getShapeProposal;

public class TensorExponentialVertex extends ProbabilisticDoubleTensor {

    private final DoubleTensorVertex a;
    private final DoubleTensorVertex b;
    private final KeanuRandom random;

    /**
     * One a or b or both driving an arbitrarily shaped tensor of Exponential
     *
     * @param shape  the desired shape of the vertex
     * @param a the a of the Exponential with either the same shape as specified for this vertex or a scalar
     * @param b the b of the Exponential with either the same shape as specified for this vertex or a scalar
     * @param random the source of randomness
     */
    public TensorExponentialVertex(int[] shape, DoubleTensorVertex a, DoubleTensorVertex b, KeanuRandom random) {

        checkParentShapes(shape, a.getValue(), b.getValue());

        this.a = a;
        this.b = b;
        this.random = random;
        setParents(a, b);
        setValue(DoubleTensor.placeHolder(shape));
    }

    /**
     * One to one constructor for mapping some shape of a and b to
     * a matching shaped exponential.
     *
     * @param a the a of the Exponential with either the same shape as specified for this vertex or a scalar
     * @param b the b of the Exponential with either the same shape as specified for this vertex or a scalar
     * @param random the source of randomness
     */
    public TensorExponentialVertex(DoubleTensorVertex a, DoubleTensorVertex b, KeanuRandom random) {
        this(getShapeProposal(a.getValue(), b.getValue()), a, b, random);
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


