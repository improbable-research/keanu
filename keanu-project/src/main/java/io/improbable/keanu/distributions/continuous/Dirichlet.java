package io.improbable.keanu.distributions.continuous;

import static io.improbable.keanu.distributions.dual.Diffs.C;
import static io.improbable.keanu.distributions.dual.Diffs.X;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Dirichlet implements ContinuousDistribution {

    private static final double EPSILON = 0.00001;
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
            DoubleTensor.ones(shape),
            concentration
        );
        final DoubleTensor gammaSamples = gamma.sample(concentration.getShape(), random);
        return normalise(gammaSamples);
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        if (Math.abs(x.sum() - 1.0) > EPSILON) {
            throw new IllegalArgumentException("Sum of values to calculate Dirichlet likelihood for must equal 1");
        }
        final double sumConcentrationLogged = concentration.minus(1.).timesInPlace(x.log()).sum();
        final double sumLogGammaConcentration = concentration.logGamma().sum();
        final double logGammaSumConcentration = org.apache.commons.math3.special.Gamma.logGamma(concentration.sum());
        return DoubleTensor.scalar(sumConcentrationLogged - sumLogGammaConcentration + logGammaSumConcentration);
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor dLogPdc = x.log()
            .minusInPlace(concentration.digamma())
            .plusInPlace(org.apache.commons.math3.special.Gamma.digamma(concentration.sum()));
        final DoubleTensor dLogPdx = concentration.minus(1).divInPlace(x);

        return new Diffs()
            .put(C, dLogPdc)
            .put(X, dLogPdx);
    }

    private DoubleTensor normalise(DoubleTensor gammaSamples) {
        double sum = gammaSamples.sum();
        return gammaSamples.div(sum);
    }
}
