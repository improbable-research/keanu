package io.improbable.keanu.distributions.continuous;

import static io.improbable.keanu.distributions.dual.Diffs.LAMBDA;
import static io.improbable.keanu.distributions.dual.Diffs.X;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Exponential implements ContinuousDistribution {

    private final DoubleTensor lambda;

    public static ContinuousDistribution withParameters(DoubleTensor lambda) {
        return new Exponential(lambda);
    }

    private Exponential(DoubleTensor lambda) {
        this.lambda = lambda;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        return random.nextDouble(shape).logInPlace().timesInPlace(lambda).unaryMinusInPlace();
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor negXMinusADivB = x.unaryMinus().divInPlace(lambda);
        final DoubleTensor negXMinusADivBMinusLogB = negXMinusADivB.minusInPlace(lambda.log());
        return negXMinusADivBMinusLogB.setWithMask(x.getLessThanMask(DoubleTensor.ZERO_SCALAR), Double.NEGATIVE_INFINITY);
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor dLogPdx = DoubleTensor.zeros(x.getShape()).plusInPlace(lambda).reciprocalInPlace().unaryMinusInPlace();
        final DoubleTensor dLogPdlambda = x.minus(lambda).divInPlace(lambda.pow(2));
        return new Diffs()
            .put(LAMBDA, dLogPdlambda)
            .put(X, dLogPdx);
    }
}
