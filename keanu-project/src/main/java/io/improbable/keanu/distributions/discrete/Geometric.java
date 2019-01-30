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
    public DoubleTensor logProb(IntegerTensor x) {
        if (!checkParameterIsValid()) {
            return DoubleTensor.create(Double.NEGATIVE_INFINITY, x.getShape());
        } else {
            return calculateLogProb(x);
        }
    }

    private DoubleTensor calculateLogProb(IntegerTensor x) {
        DoubleTensor xAsDouble = x.toDouble();
        DoubleTensor oneMinusP = p.unaryMinus().plusInPlace(1.0);
        DoubleTensor results = xAsDouble.minusInPlace(1.0).timesInPlace(oneMinusP.logInPlace()).plusInPlace(p.log());

        return setProbToZeroForInvalidX(xAsDouble, results);
    }

    private DoubleTensor setProbToZeroForInvalidX(DoubleTensor x, DoubleTensor results) {
        DoubleTensor invalidX = results.getGreaterThanOrEqualToMask(DoubleTensor.create(1.0, x.getShape()));

        return results.setWithMaskInPlace(invalidX, Double.NEGATIVE_INFINITY);
    }

    private boolean checkParameterIsValid() {
        return p.greaterThan(0.0).allTrue() && p.lessThan(1.0).allTrue();
    }
}
