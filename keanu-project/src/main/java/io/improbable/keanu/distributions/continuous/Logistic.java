package io.improbable.keanu.distributions.continuous;

import static io.improbable.keanu.distributions.dual.Diffs.MU;
import static io.improbable.keanu.distributions.dual.Diffs.S;
import static io.improbable.keanu.distributions.dual.Diffs.X;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

/**
 * @see "Computer Generation of Statistical Distributions
 * by Richard Saucier,
 * ARL-TR-2168 March 2000,
 * 5.1.14 page 27"
 */
public class Logistic implements ContinuousDistribution {

    private final DoubleTensor location;
    private final DoubleTensor scale;

    /**
     * @param location shifts the distribution
     * @param scale    stretches/shrinks the distribution
     */
    public static ContinuousDistribution withParameters(DoubleTensor location, DoubleTensor scale) {
        return new Logistic(location, scale);
    }

    private Logistic(DoubleTensor location, DoubleTensor scale) {
        this.location = location;
        this.scale = scale;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        return random.nextDouble(shape).reciprocalInPlace().minusInPlace(1).logInPlace().timesInPlace(location.minus(scale));
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor xMinusAOverB = x.minus(location).divInPlace(scale);
        final DoubleTensor ln1OverB = scale.reciprocal().logInPlace();

        return xMinusAOverB.plus(ln1OverB).minusInPlace(
            xMinusAOverB.expInPlace().plusInPlace(1).logInPlace().timesInPlace(2)
        );
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor expAOverB = location.div(scale).expInPlace();
        final DoubleTensor expXOverB = x.div(scale).expInPlace();
        final DoubleTensor expPlus = expAOverB.plus(expXOverB);
        final DoubleTensor bTimesExpAOverB = expAOverB.times(scale);
        final DoubleTensor bTimesExpXOverB = expXOverB.times(scale);

        final DoubleTensor dLogPdlocation = expXOverB.minus(expAOverB).divInPlace(scale.times(expPlus));
        final DoubleTensor dLogPdx = expAOverB.minus(expXOverB).divInPlace(bTimesExpAOverB.plus(bTimesExpXOverB));

        final DoubleTensor numeratorPartOne = location.times(expXOverB).plusInPlace(x.times(expAOverB)).plusInPlace(
            location.times(expAOverB.unaryMinus())
        );
        final DoubleTensor numeratorPartTwo = bTimesExpAOverB.plus(bTimesExpXOverB).minusInPlace(x.times(expXOverB));
        final DoubleTensor denominator = scale.pow(2).timesInPlace(expPlus);

        final DoubleTensor dLogPdscale = numeratorPartOne.plus(numeratorPartTwo).divInPlace(denominator).unaryMinusInPlace();

        return new Diffs()
            .put(MU, dLogPdlocation)
            .put(S, dLogPdscale)
            .put(X, dLogPdx);
    }

}