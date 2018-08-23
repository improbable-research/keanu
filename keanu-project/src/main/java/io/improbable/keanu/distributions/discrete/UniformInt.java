package io.improbable.keanu.distributions.discrete;

import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.distributions.IntegerSupport;
import io.improbable.keanu.distributions.Support;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class UniformInt implements DiscreteDistribution {

    private final IntegerTensor xMin;
    private final IntegerTensor xMax;

    public static DiscreteDistribution withParameters(IntegerTensor xMin, IntegerTensor xMax) {
        return new UniformInt(xMin, xMax);
    }

    private UniformInt(IntegerTensor xMin, IntegerTensor xMax) {
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
        DoubleTensor xDouble = x.toDouble();

        DoubleTensor logOfWithinBounds = maxBound.minus(minBound).logInPlace().unaryMinusInPlace();
        logOfWithinBounds = logOfWithinBounds.setWithMaskInPlace(xDouble.getGreaterThanOrEqualToMask(maxBound), Double.NEGATIVE_INFINITY);
        logOfWithinBounds = logOfWithinBounds.setWithMaskInPlace(xDouble.getLessThanMask(minBound), Double.NEGATIVE_INFINITY);

        return logOfWithinBounds;
    }

    @Override
    public Support<IntegerTensor> getSupport() {
        return new IntegerSupport(xMin, xMax, xMin.getShape());
    }
}
