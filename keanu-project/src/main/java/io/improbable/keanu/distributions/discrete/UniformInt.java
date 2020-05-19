package io.improbable.keanu.distributions.discrete;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerPlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;

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
    public IntegerTensor sample(long[] shape, KeanuRandom random) {
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
        logOfWithinBounds = logOfWithinBounds.setWithMaskInPlace(xDouble.greaterThanOrEqualToMask(maxBound), Double.NEGATIVE_INFINITY);
        logOfWithinBounds = logOfWithinBounds.setWithMaskInPlace(xDouble.lessThanMask(minBound), Double.NEGATIVE_INFINITY);

        return logOfWithinBounds;
    }

    public static DoubleVertex logProbOutput(IntegerPlaceholderVertex x, IntegerPlaceholderVertex xMin, IntegerPlaceholderVertex xMax) {
        DoubleVertex maxBound = xMax.toDouble();
        DoubleVertex minBound = xMin.toDouble();
        DoubleVertex xDouble = x.toDouble();

        DoubleVertex logOfWithinBounds = maxBound.minus(minBound).log().unaryMinus();
        logOfWithinBounds = logOfWithinBounds.setWithMask(xDouble.greaterThanOrEqualToMask(maxBound), Double.NEGATIVE_INFINITY);
        logOfWithinBounds = logOfWithinBounds.setWithMask(xDouble.lessThanMask(minBound), Double.NEGATIVE_INFINITY);

        return logOfWithinBounds;
    }
}
