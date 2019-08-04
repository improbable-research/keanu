package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;

import static io.improbable.keanu.distributions.hyperparam.Diffs.LAMBDA;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;

public class Exponential implements ContinuousDistribution {

    private final DoubleTensor lambda;

    public static ContinuousDistribution withParameters(DoubleTensor lambda) {
        return new Exponential(lambda);
    }

    private Exponential(DoubleTensor lambda) {
        this.lambda = lambda;
    }

    @Override
    public DoubleTensor sample(long[] shape, KeanuRandom random) {
        return random.nextDouble(shape).logInPlace().timesInPlace(lambda).unaryMinusInPlace();
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor negXMinusADivB = x.unaryMinus().divInPlace(lambda);
        final DoubleTensor negXMinusADivBMinusLogB = negXMinusADivB.minusInPlace(lambda.log());
        return negXMinusADivBMinusLogB.setWithMask(x.lessThanMask(DoubleTensor.scalar(0.0)), Double.NEGATIVE_INFINITY);
    }

    public static DoubleVertex logProbOutput(DoublePlaceholderVertex x, DoublePlaceholderVertex lambda) {
        final DoubleVertex negXMinusADivB = x.unaryMinus().div(lambda);
        final DoubleVertex negXMinusADivBMinusLogB = negXMinusADivB.minus(lambda.log());
        return negXMinusADivBMinusLogB.setWithMask(x.lessThanMask(0.), Double.NEGATIVE_INFINITY);
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor dLogPdx = DoubleTensor.zeros(x.getShape()).minusInPlace(lambda).reciprocalInPlace();
        final DoubleTensor dLogPdlambda = x.minus(lambda).divInPlace(lambda.pow(2));
        return new Diffs()
            .put(LAMBDA, dLogPdlambda)
            .put(X, dLogPdx);
    }
}
