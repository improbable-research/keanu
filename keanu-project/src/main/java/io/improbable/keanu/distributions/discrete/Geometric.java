package io.improbable.keanu.distributions.discrete;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LogProbGraph.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.LogProbGraph.IntegerPlaceHolderVertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

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

    public static DoubleVertex logProbOutput(IntegerPlaceHolderVertex k, DoublePlaceholderVertex p) {
        ConstantDoubleVertex zero = ConstantVertex.of(DoubleTensor.zeros(k.getShape()));
        ConstantDoubleVertex one = ConstantVertex.of(DoubleTensor.ones(k.getShape()));

        BooleanVertex parameterIsValid = p.greaterThan(zero).and(p.lessThan(one));
        return If.isTrue(parameterIsValid)
            .then(calculateLogProb(k, p))
            .orElse(ConstantVertex.of(DoubleTensor.create(Double.NEGATIVE_INFINITY, k.getShape())));
    }

    private DoubleTensor calculateLogProb(IntegerTensor k) {
        DoubleTensor kAsDouble = k.toDouble();
        DoubleTensor oneMinusP = p.unaryMinus().plusInPlace(1.0);
        DoubleTensor results = kAsDouble.minusInPlace(1.0).timesInPlace(oneMinusP.logInPlace()).plusInPlace(p.log());

        return setProbToZeroForInvalidK(k, results);
    }

    private static DoubleVertex calculateLogProb(IntegerVertex k, DoubleVertex p) {
        DoubleVertex kAsDouble = k.toDouble();
        DoubleVertex oneMinusP = p.unaryMinus().plus(1.0);
        DoubleVertex results = kAsDouble.minus(1.0).times(oneMinusP.log()).plus(p.log());

        return setProbToZeroForInvalidK(k, results);
    }

    private DoubleTensor setProbToZeroForInvalidK(IntegerTensor k, DoubleTensor results) {
        IntegerTensor invalidK = k.getLessThanMask(IntegerTensor.create(1, k.getShape()));

        return results.setWithMaskInPlace(invalidK.toDouble(), Double.NEGATIVE_INFINITY);
    }

    private static DoubleVertex setProbToZeroForInvalidK(IntegerVertex k, DoubleVertex results) {
        DoubleVertex invalidK = k.toDouble().toLessThanMask(1.);

        return results.setWithMask(invalidK, Double.NEGATIVE_INFINITY);
    }

    private boolean checkParameterIsValid() {
        return p.greaterThan(0.0).allTrue() && p.lessThan(1.0).allTrue();
    }

}
