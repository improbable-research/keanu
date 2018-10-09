package io.improbable.keanu.distributions.continuous;

import static io.improbable.keanu.distributions.hyperparam.Diffs.MU;
import static io.improbable.keanu.distributions.hyperparam.Diffs.S;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Logistic implements ContinuousDistribution {

    private final DoubleTensor mu;
    private final DoubleTensor s;

    /**
     * @param mu     location parameter (any real number)
     * @param s      scale parameter (b greater than 0)
     * @return       a new ContinuousDistribution object
     */
    public static ContinuousDistribution withParameters(DoubleTensor mu, DoubleTensor s) {
        return new Logistic(mu, s);
    }

    private Logistic(DoubleTensor mu, DoubleTensor s) {
        this.mu = mu;
        this.s = s;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        return random.nextDouble(shape).reciprocalInPlace().minusInPlace(1).logInPlace().timesInPlace(mu.minus(s));
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor xMinusAOverB = x.minus(mu).divInPlace(s);
        final DoubleTensor ln1OverB = s.reciprocal().logInPlace();

        return xMinusAOverB.plus(ln1OverB).minusInPlace(
            xMinusAOverB.expInPlace().plusInPlace(1).logInPlace().timesInPlace(2)
        );
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor expAOverB = mu.div(s).expInPlace();
        final DoubleTensor expXOverB = x.div(s).expInPlace();
        final DoubleTensor expPlus = expAOverB.plus(expXOverB);
        final DoubleTensor bTimesExpAOverB = expAOverB.times(s);
        final DoubleTensor bTimesExpXOverB = expXOverB.times(s);

        final DoubleTensor dLogPdmu = expXOverB.minus(expAOverB).divInPlace(s.times(expPlus));
        final DoubleTensor dLogPdx = expAOverB.minus(expXOverB).divInPlace(bTimesExpAOverB.plus(bTimesExpXOverB));

        final DoubleTensor numeratorPartOne = mu.times(expXOverB).plusInPlace(x.times(expAOverB)).plusInPlace(
            mu.times(expAOverB.unaryMinus())
        );
        final DoubleTensor numeratorPartTwo = bTimesExpAOverB.plus(bTimesExpXOverB).minusInPlace(x.times(expXOverB));
        final DoubleTensor denominator = s.pow(2).timesInPlace(expPlus);

        final DoubleTensor dLogPds = numeratorPartOne.plus(numeratorPartTwo).divInPlace(denominator).unaryMinusInPlace();

        return new Diffs()
            .put(MU, dLogPdmu)
            .put(S, dLogPds)
            .put(X, dLogPdx);
    }
}
