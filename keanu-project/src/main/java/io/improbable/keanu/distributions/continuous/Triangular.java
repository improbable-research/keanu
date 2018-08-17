package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

/**
 * @see "Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.1.24 page 37"
 */
public class Triangular implements ContinuousDistribution {

    private final DoubleTensor xMin;
    private final DoubleTensor xMax;
    private final DoubleTensor mode;

    /**
     * @param xMin minimum value of random variable x
     * @param xMax maximum value of random variable x
     * @param mode location of mode
     * @return an instance of {@link ContinuousDistribution}
     */
    public static ContinuousDistribution withParameters(DoubleTensor xMin, DoubleTensor xMax, DoubleTensor mode) {
        return new Triangular(xMin, xMax, mode);
    }

    private Triangular(DoubleTensor xMin, DoubleTensor xMax, DoubleTensor mode) {
        this.xMin = xMin;
        this.xMax = xMax;
        this.mode = mode;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        final DoubleTensor p = random.nextDouble(shape);
        final DoubleTensor q = p.unaryMinus().plusInPlace(1);
        final DoubleTensor range = xMax.minus(xMin);

        final DoubleTensor conditional = (mode.minus(xMin)).divInPlace(xMax.minus(xMin));

        final DoubleTensor lessThan = xMin.plus((range.times(mode.minus(xMin).timesInPlace(p))).sqrtInPlace());
        final DoubleTensor greaterThan = xMax.minus((range.timesInPlace(xMax.minus(mode).timesInPlace(q))).sqrtInPlace());

        final DoubleTensor lessThanMask = p.getLessThanOrEqualToMask(conditional);
        final DoubleTensor greaterThanMask = p.getGreaterThanMask(conditional);

        return (lessThan.timesInPlace(lessThanMask).plusInPlace(greaterThan.timesInPlace(greaterThanMask)));
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor range = xMax.minus(xMin);

        final DoubleTensor conditionalFirstHalf = x.getGreaterThanMask(xMin);
        final DoubleTensor conditionalSecondHalf = x.getLessThanMask(mode);
        final DoubleTensor conditionalAnd = conditionalFirstHalf.timesInPlace(conditionalSecondHalf);
        final DoubleTensor conditionalResult = range.reciprocal().timesInPlace(2).timesInPlace(x.minus(xMin)).divInPlace(mode.minus(xMin));

        final DoubleTensor elseIfConditionalFirstHalf = x.getGreaterThanMask(mode);
        final DoubleTensor elseIfConditionalSecondHalf = x.getLessThanOrEqualToMask(xMax);
        final DoubleTensor elseIfConditionalAnd = elseIfConditionalFirstHalf.timesInPlace(elseIfConditionalSecondHalf);
        final DoubleTensor elseIfConditionalResult = range.reciprocalInPlace().timesInPlace(2).timesInPlace(xMax.minus(x)).divInPlace(xMax.minus(mode));

        return (conditionalResult.timesInPlace(conditionalAnd).plusInPlace(elseIfConditionalResult.timesInPlace(elseIfConditionalAnd))).logInPlace();
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        throw new UnsupportedOperationException();
    }

}