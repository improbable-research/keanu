package io.improbable.keanu.distributions.discrete;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

/**
 * Implements a Geometric Random Distribution.  More details can be found at:
 * https://en.wikipedia.org/wiki/Geometric_distribution
 */
public class Geometric implements DiscreteDistribution {

    private final DoubleTensor p;

    public static DiscreteDistribution withParameters(DoubleTensor p) {
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

    private DoubleTensor calculateLogProb(IntegerTensor k) {
        DoubleTensor xAsDouble = k.toDouble();
        DoubleTensor oneMinusP = p.unaryMinus().plusInPlace(1.0);
        DoubleTensor results = xAsDouble.minusInPlace(1.0).timesInPlace(oneMinusP.logInPlace()).plusInPlace(p.log());

        return setProbToZeroForInvalidK(k, results);
    }

    private DoubleTensor setProbToZeroForInvalidK(IntegerTensor k, DoubleTensor results) {
        IntegerTensor invalidX = k.getLessThanMask(IntegerTensor.create(1, k.getShape()));

        return results.setWithMaskInPlace(invalidX.toDouble(), Double.NEGATIVE_INFINITY);
    }

    private boolean checkParameterIsValid() {
        return p.greaterThan(0.0).allTrue() && p.lessThan(1.0).allTrue();
    }
}
