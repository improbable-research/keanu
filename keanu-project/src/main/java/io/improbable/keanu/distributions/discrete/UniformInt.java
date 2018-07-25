package io.improbable.keanu.distributions.discrete;

import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class UniformInt implements DiscreteDistribution {

    private final IntegerTensor xMin;
    private final IntegerTensor xMax;

    /**
     * @param xMin   minimum x value
     * @param xMax   maximum x value
     */
    // TODO: package private
    public UniformInt(IntegerTensor xMin, IntegerTensor xMax) {
        this.xMin = xMin;
        this.xMax = xMax;
    }

    @Override
    public IntegerTensor sample(int[] shape, KeanuRandom random) {
        DoubleTensor minDouble = xMin.toDouble();
        DoubleTensor delta = xMax.toDouble().minus(minDouble);
        DoubleTensor randoms = random.nextDouble(shape);

        return delta.timesInPlace(randoms).plusInPlace(minDouble).toInteger();
    }

    @Override
    public DoubleTensor logProb(IntegerTensor x) {
        DoubleTensor maxBound = xMax.toDouble();
        DoubleTensor minBound = xMin.toDouble();
        DoubleTensor xAsDouble = x.toDouble();

        DoubleTensor logOfWithinBounds = maxBound.minus(minBound).logInPlace().unaryMinusInPlace();
        logOfWithinBounds = logOfWithinBounds.setWithMaskInPlace(xAsDouble.getGreaterThanMask(maxBound), Double.NEGATIVE_INFINITY);
        logOfWithinBounds = logOfWithinBounds.setWithMaskInPlace(xAsDouble.getLessThanOrEqualToMask(minBound), Double.NEGATIVE_INFINITY);

        return logOfWithinBounds;
    }
}
