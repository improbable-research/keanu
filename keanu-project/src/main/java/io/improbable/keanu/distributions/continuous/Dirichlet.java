package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static io.improbable.keanu.distributions.dual.Diffs.C;
import static io.improbable.keanu.distributions.dual.Diffs.X;

public class Dirichlet implements ContinuousDistribution {

    private final DoubleTensor concentration;

    public static ContinuousDistribution withParameters(DoubleTensor concentration) {
        return new Dirichlet(concentration);
    }

    private Dirichlet(DoubleTensor concentration) {
        this.concentration = concentration;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
       final ContinuousDistribution gamma = Gamma.withParameters(
            concentration,
            DoubleTensor.ones(shape),
            DoubleTensor.zeros(shape)
        );
        final DoubleTensor gammaSamples = gamma.sample(concentration.getShape(), random);
        return normalise(gammaSamples);
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor sumConcentrationLogged = x.log().times(concentration.minus(1.).sum());
        final double sumLogGammaConcentration = concentration.apply(org.apache.commons.math3.special.Gamma::gamma).log().sum();
        final double logGammaSumConcentration = Math.log(org.apache.commons.math3.special.Gamma.gamma(concentration.sum()));
        return sumConcentrationLogged.minus(sumLogGammaConcentration).plus(logGammaSumConcentration);
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor dLogPdc = x.log().minus(concentration.apply(org.apache.commons.math3.special.Gamma::digamma)).plus(org.apache.commons.math3.special.Gamma.digamma(concentration.sum()));
        final DoubleTensor dLogPdx = concentration.minus(1).div(x);

        return new Diffs()
            .put(C, dLogPdc)
            .put(X, dLogPdx);
    }

    private DoubleTensor normalise(DoubleTensor gammaSamples) {
        double sum = gammaSamples.sum();
        return gammaSamples.div(sum);
    }
}
