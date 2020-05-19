package io.improbable.keanu.distributions.discrete;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerPlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;

/**
 * Implements a Geometric Random Distribution.  More details can be found at:
 * https://en.wikipedia.org/wiki/Geometric_distribution
 */
public class Geometric implements DiscreteDistribution {

    private final DoubleTensor p;

    public static Geometric withParameters(DoubleTensor p) {
        return new Geometric(p);
    }

    private Geometric(DoubleTensor p) {
        this.p = p;
    }

    @Override
    public IntegerTensor sample(long[] shape, KeanuRandom random) {
        DoubleTensor numerator = random.nextDouble(shape).logInPlace();
        DoubleTensor denominator = p.unaryMinus().plusInPlace(1.0).logInPlace();

        return numerator.divInPlace(denominator).floorInPlace().toInteger().plusInPlace(1);
    }

    @Override
    public DoubleTensor logProb(IntegerTensor k) {
        if (!checkParameterIsValid()) {
            return DoubleTensor.create(Double.NEGATIVE_INFINITY, k.getShape());
        } else {
            return calculateLogProb(k);
        }
    }

    public static DoubleVertex logProbOutput(IntegerPlaceholderVertex k, DoublePlaceholderVertex p) {
        DoubleVertex zeroes = ConstantVertex.of(DoubleTensor.zeros(k.getShape()));
        DoubleVertex ones = ConstantVertex.of(DoubleTensor.ones(k.getShape()));
        DoubleVertex parameterIsInvalidMask = p.greaterThanMask(zeroes)
            .times(p.lessThanMask(ones))
            .unaryMinus()
            .plus(ones);
        return calculateLogProb(k, p).setWithMask(parameterIsInvalidMask, Double.NEGATIVE_INFINITY);
    }

    private DoubleTensor calculateLogProb(IntegerTensor k) {
        DoubleTensor kAsDouble = k.toDouble();
        DoubleTensor oneMinusP = p.reverseMinus(1.0);
        DoubleTensor results = oneMinusP.safeLogTimesInPlace(kAsDouble.minusInPlace(1.0)).plusInPlace(p.log());

        return setProbToZeroForInvalidK(k, results);
    }

    private static DoubleVertex calculateLogProb(IntegerVertex k, DoubleVertex p) {
        DoubleVertex kAsDouble = k.toDouble();
        DoubleVertex oneMinusP = p.unaryMinus().plus(1.0);
        DoubleVertex results = kAsDouble.minus(1.0).times(oneMinusP.log()).plus(p.log());

        return setProbToZeroForInvalidK(k, results);
    }

    private DoubleTensor setProbToZeroForInvalidK(IntegerTensor k, DoubleTensor results) {
        IntegerTensor invalidK = k.lessThanMask(IntegerTensor.create(1, k.getShape()));

        return results.setWithMaskInPlace(invalidK.toDouble(), Double.NEGATIVE_INFINITY);
    }

    private static DoubleVertex setProbToZeroForInvalidK(IntegerVertex k, DoubleVertex results) {
        DoubleVertex invalidK = k.toDouble().lessThanMask(1.);

        return results.setWithMask(invalidK, Double.NEGATIVE_INFINITY);
    }

    private boolean checkParameterIsValid() {
        return p.greaterThan(0.0).allTrue().scalar() && p.lessThan(1.0).allTrue().scalar();
    }

    public DoubleTensor[] dLogProb(IntegerTensor k, boolean wrtP) {
        DoubleTensor[] result = new DoubleTensor[1];

        if (wrtP) {
            DoubleTensor diffWhereValid = p.reciprocal().minusInPlace(k.toDouble().minusInPlace(1.0).divInPlace(p.reverseMinus(1.0)));
            result[0] = diffWhereValid.where(k.greaterThanOrEqual(1), DoubleTensor.scalar(0.0));
        }

        return result;
    }

}
