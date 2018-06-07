package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class UniformIntVertex extends ProbabilisticInteger {

    private IntegerVertex min;
    private IntegerVertex max;

    /**
     * @param shape tensor shape of value
     * @param min   The inclusive lower bound.
     * @param max   The exclusive upper bound.
     */
    public UniformIntVertex(int[] shape, IntegerVertex min, IntegerVertex max) {

        checkTensorsMatchNonScalarShapeOrAreScalar(shape, min.getShape(), max.getShape());

        this.min = min;
        this.max = max;
        setParents(min, max);
        setValue(IntegerTensor.placeHolder(shape));
    }

    public UniformIntVertex(int[] shape, int min, int max) {
        this(shape, new ConstantIntegerVertex(min), new ConstantIntegerVertex(max));
    }

    public UniformIntVertex(int[] shape, IntegerTensor min, IntegerTensor max) {
        this(shape, new ConstantIntegerVertex(min), new ConstantIntegerVertex(max));
    }

    public UniformIntVertex(int[] shape, IntegerVertex min, int max) {
        this(shape, min, new ConstantIntegerVertex(max));
    }

    public UniformIntVertex(int[] shape, int min, IntegerVertex max) {
        this(shape, new ConstantIntegerVertex(min), max);
    }

    public UniformIntVertex(IntegerVertex min, IntegerVertex max) {
        this(checkHasSingleNonScalarShapeOrAllScalar(min.getShape(), max.getShape()), min, max);
    }

    public UniformIntVertex(IntegerVertex min, int max) {
        this(min.getShape(), min, new ConstantIntegerVertex(max));
    }

    public UniformIntVertex(int min, IntegerVertex max) {
        this(max.getShape(), new ConstantIntegerVertex(min), max);
    }

    public UniformIntVertex(int min, int max) {
        this(Tensor.SCALAR_SHAPE, new ConstantIntegerVertex(min), new ConstantIntegerVertex(max));
    }

    public Vertex<IntegerTensor> getMin() {
        return min;
    }

    public Vertex<IntegerTensor> getMax() {
        return max;
    }

    @Override
    public double logPmf(IntegerTensor value) {

        DoubleTensor maxBound = max.getValue().toDouble();
        DoubleTensor minBound = min.getValue().toDouble();
        DoubleTensor x = value.toDouble();

        DoubleTensor logOfWithinBounds = maxBound.minus(minBound).logInPlace().unaryMinusInPlace();
        logOfWithinBounds = logOfWithinBounds.setWithMaskInPlace(x.getGreaterThanMask(maxBound), Double.NEGATIVE_INFINITY);
        logOfWithinBounds = logOfWithinBounds.setWithMaskInPlace(x.getLessThanOrEqualToMask(minBound), Double.NEGATIVE_INFINITY);

        return logOfWithinBounds.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPmf(IntegerTensor value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {

        DoubleTensor delta = max.getValue().minus(min.getValue()).toDouble();
        DoubleTensor randoms = random.nextDouble(delta.getShape());

        return delta.timesInPlace(randoms).toInteger().plusInPlace(min.getValue());
    }
}
