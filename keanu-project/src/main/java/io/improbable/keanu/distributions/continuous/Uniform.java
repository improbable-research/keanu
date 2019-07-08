package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

/**
 * Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.1.8 page 48
 */
public class Uniform implements ContinuousDistribution {

    private final DoubleTensor xMin;
    private final DoubleTensor xMax;

    /**
     * @param xMin minimum x value
     * @param xMax maximum x value
     * @return a new ContinuousDistribution object
     */
    public static ContinuousDistribution withParameters(DoubleTensor xMin, DoubleTensor xMax) {
        return new Uniform(xMin, xMax);
    }

    private Uniform(DoubleTensor xMin, DoubleTensor xMax) {
        this.xMin = xMin;
        this.xMax = xMax;
    }

    @Override
    public DoubleTensor sample(long[] shape, KeanuRandom random) {
        return random.nextDouble(shape).timesInPlace(xMax.minus(xMin)).plusInPlace(xMin);
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {

        DoubleTensor logOfWithinBounds = xMax.minus(xMin).logInPlace().unaryMinusInPlace();
        logOfWithinBounds = logOfWithinBounds.setWithMaskInPlace(x.greaterThanOrEqualToMask(xMax), Double.NEGATIVE_INFINITY);
        logOfWithinBounds = logOfWithinBounds.setWithMaskInPlace(x.lessThanMask(xMin), Double.NEGATIVE_INFINITY);

        return logOfWithinBounds;
    }

    public static DoubleVertex logProbOutput(DoublePlaceholderVertex x, DoublePlaceholderVertex xMin, DoublePlaceholderVertex xMax) {

        DoubleVertex logOfWithinBounds = xMax.minus(xMin).log().unaryMinus();
        logOfWithinBounds = logOfWithinBounds.setWithMask(x.greaterThanOrEqualToMask(xMax), Double.NEGATIVE_INFINITY);
        logOfWithinBounds = logOfWithinBounds.setWithMask(x.lessThanMask(xMin), Double.NEGATIVE_INFINITY);

        return logOfWithinBounds;
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        throw new UnsupportedOperationException();
    }
}
