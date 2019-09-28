package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;

import static io.improbable.keanu.distributions.hyperparam.Diffs.MU;
import static io.improbable.keanu.distributions.hyperparam.Diffs.S;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;

public class Logistic implements ContinuousDistribution {

    private final DoubleTensor mu;
    private final DoubleTensor s;

    /**
     * @param mu location parameter (any real number)
     * @param s  scale parameter (b greater than 0)
     * @return a new ContinuousDistribution object
     */
    public static Logistic withParameters(DoubleTensor mu, DoubleTensor s) {
        return new Logistic(mu, s);
    }

    private Logistic(DoubleTensor mu, DoubleTensor s) {
        this.mu = mu;
        this.s = s;
    }

    @Override
    public DoubleTensor sample(long[] shape, KeanuRandom random) {
        return random.nextDouble(shape).reciprocalInPlace().minusInPlace(1.0).logInPlace().timesInPlace(s).plusInPlace(mu);
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor muMinusXOverS = mu.minus(x).divInPlace(s);

        return muMinusXOverS.minus(s.log()).minusInPlace(
            muMinusXOverS.exp().plusInPlace(1.).logInPlace().timesInPlace(2.)
        );
    }

    public static DoubleVertex logProbOutput(DoublePlaceholderVertex x, DoublePlaceholderVertex mu, DoublePlaceholderVertex s) {
        final DoubleVertex xMinusAOverB = mu.minus(x).div(s);
        final DoubleVertex sLog = s.log();

        return xMinusAOverB.minus(sLog).minus(
            xMinusAOverB.exp().plus(1.).log().times(2.)
        );
    }

    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor muMinusXOverS = mu.minus(x).divInPlace(s);
        final DoubleTensor expMuMinusXOverS = muMinusXOverS.exp();
        final DoubleTensor A = expMuMinusXOverS.times(2.0).div(expMuMinusXOverS.plus(1));

        final DoubleTensor dLogPdx = A.minus(1).div(s);
        final DoubleTensor dLogPdmu = dLogPdx.unaryMinus();
        final DoubleTensor dLogPds = A.times(muMinusXOverS).minus(muMinusXOverS).minus(1.0).div(s);

        return new Diffs()
            .put(MU, dLogPdmu)
            .put(S, dLogPds)
            .put(X, dLogPdx);
    }
}
