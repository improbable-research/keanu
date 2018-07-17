package io.improbable.keanu.distributions.continuous;

import static io.improbable.keanu.distributions.dual.Diffs.A;
import static io.improbable.keanu.distributions.dual.Diffs.B;
import static io.improbable.keanu.distributions.dual.Diffs.X;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Exponential implements ContinuousDistribution {

    private final DoubleTensor location;
    private final DoubleTensor lambda;

    public static ContinuousDistribution withParameters(DoubleTensor location, DoubleTensor lambda) {
        return new Exponential(location, lambda);
    }

    private Exponential(DoubleTensor location, DoubleTensor lambda) {
        this.location = location;
        this.lambda = lambda;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        return location.minus(random.nextDouble(shape).logInPlace().timesInPlace(lambda));
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor negXMinusADivB = x.minus(location).unaryMinusInPlace().divInPlace(lambda);
        final DoubleTensor negXMinusADivBMinusLogB = negXMinusADivB.minusInPlace(lambda.log());
        return negXMinusADivBMinusLogB.setWithMask(x.getLessThanMask(location), Double.NEGATIVE_INFINITY);
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor dPda = lambda.reciprocal();
        final DoubleTensor dPdb = x.minus(location).minusInPlace(lambda).divInPlace(lambda.pow(2));
        return new Diffs()
            .put(A, dPda)
            .put(B, dPdb)
            .put(X, dPda.unaryMinus());
    }
}
