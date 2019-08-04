package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.tensor.bool.BooleanVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;

import static io.improbable.keanu.distributions.hyperparam.Diffs.C;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;

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
    public DoubleTensor sample(long[] shape, KeanuRandom random) {
        final ContinuousDistribution gamma = Gamma.withParameters(
            DoubleTensor.ones(shape),
            concentration
        );
        final DoubleTensor gammaSamples = gamma.sample(concentration.getShape(), random);
        return normalise(gammaSamples);
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        if (Math.abs(x.sumNumber() - 1.0) > EPSILON) {
            throw new IllegalArgumentException("Sum of values to calculate Dirichlet likelihood for must equal 1");
        }
        final double sumConcentrationLogged = concentration.minus(1.).timesInPlace(x.log()).sumNumber();
        final double sumLogGammaConcentration = concentration.logGamma().sumNumber();
        final double logGammaSumConcentration = org.apache.commons.math3.special.Gamma.logGamma(concentration.sumNumber());
        return DoubleTensor.scalar(sumConcentrationLogged - sumLogGammaConcentration + logGammaSumConcentration);
    }

    public static DoubleVertex logProbOutput(DoublePlaceholderVertex x, DoublePlaceholderVertex concentration) {
        final BooleanVertex xMinusOneIsLessThanOrEqualToEpsilon = x
            .sum().minus(1.).abs().lessThanOrEqual(ConstantVertex.of(EPSILON));
        xMinusOneIsLessThanOrEqualToEpsilon.assertTrue("Sum of values to calculate Dirichlet likelihood for must equal 1");

        final DoubleVertex sumConcentrationLogged = concentration.minus(1.).times(x.log()).sum();
        final DoubleVertex sumLogGammaConcentration = concentration.logGamma().sum();
        final DoubleVertex logGammaSumConcentration = concentration.sum().logGamma();
        return sumConcentrationLogged.minus(sumLogGammaConcentration).plus(logGammaSumConcentration);
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor dLogPdc = x.log()
            .minusInPlace(concentration.digamma())
            .plusInPlace(org.apache.commons.math3.special.Gamma.digamma(concentration.sumNumber()));
        final DoubleTensor dLogPdx = concentration.minus(1).divInPlace(x);

        return new Diffs()
            .put(C, dLogPdc)
            .put(X, dLogPdx);
    }

    private DoubleTensor normalise(DoubleTensor gammaSamples) {
        double sum = gammaSamples.sumNumber();
        return gammaSamples.div(sum);
    }
}
