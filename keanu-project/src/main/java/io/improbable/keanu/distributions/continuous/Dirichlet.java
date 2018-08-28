package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.Distribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static io.improbable.keanu.distributions.dual.Diffs.C;
import static io.improbable.keanu.distributions.dual.Diffs.X;

public class Dirichlet implements ContinuousDistribution {

    private static final double EPSILON =  0.00001;
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
        final double sumLogGammaConcentration = concentration.apply(org.apache.commons.math3.special.Gamma::gamma).logInPlace().sum();
        final double logGammaSumConcentration = Math.log(org.apache.commons.math3.special.Gamma.gamma(concentration.sum()));
        return DoubleTensor.scalar(sumConcentrationLogged - sumLogGammaConcentration + logGammaSumConcentration);
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor dLogPdc = x.log()
            .minusInPlace(concentration.apply(org.apache.commons.math3.special.Gamma::digamma))
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

    @Override
    public DoubleTensor computeKLDivergence(Distribution<DoubleTensor> q) {
        if (q instanceof Dirichlet) {
            DoubleTensor qConcentration = ((Dirichlet) q).concentration;

            DoubleTensor concentrationDiff = qConcentration.minus(concentration);

            DoubleTensor digammaPConcentration = concentration.apply(org.apache.commons.math3.special.Gamma::digamma);
            DoubleTensor digammaSumP = concentration.sum(-1).applyInPlace(org.apache.commons.math3.special.Gamma::digamma);
            DoubleTensor digammaDiff = digammaPConcentration.minusInPlace(digammaSumP);

            return concentrationDiff
                .timesInPlace(digammaDiff)
                .sum(-1)
                .minusInPlace(logBetaX(concentration))
                .plusInPlace(logBetaX(qConcentration));
        } else {
            return ContinuousDistribution.super.computeKLDivergence(q);
        }
    }

    private DoubleTensor logBetaX(DoubleTensor x) {
        DoubleTensor logProdGammaX = x.sum(-1).applyInPlace(org.apache.commons.math3.special.Gamma::logGamma);

        DoubleTensor sumX = x.sum(-1);
        DoubleTensor logGammaSumX = sumX.applyInPlace(org.apache.commons.math3.special.Gamma::logGamma);

        return logProdGammaX.minusInPlace(logGammaSumX);
    }
}
