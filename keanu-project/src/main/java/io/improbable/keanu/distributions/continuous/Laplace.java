package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import static io.improbable.keanu.distributions.hyperparam.Diffs.BETA;
import static io.improbable.keanu.distributions.hyperparam.Diffs.MU;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;

public class Laplace implements ContinuousDistribution {

    private final DoubleTensor mu;
    private final DoubleTensor beta;

    /**
     * @param mu   location
     * @param beta shape
     * @return a new ContinuousDistribution object
     */
    public static ContinuousDistribution withParameters(DoubleTensor mu, DoubleTensor beta) {
        return new Laplace(mu, beta);
    }

    private Laplace(DoubleTensor mu, DoubleTensor beta) {
        this.mu = mu;
        this.beta = beta;
    }

    @Override
    public DoubleTensor sample(long[] shape, KeanuRandom random) {
        Tensor.FlattenedView<Double> muWrapped = mu.getFlattenedView();
        Tensor.FlattenedView<Double> betaWrapped = beta.getFlattenedView();

        int length = TensorShape.getLengthAsInt(shape);
        double[] samples = new double[length];
        for (int i = 0; i < length; i++) {
            samples[i] = sample(muWrapped.getOrScalar(i), betaWrapped.getOrScalar(i), random);
        }

        return DoubleTensor.create(samples, shape);
    }

    private static double sample(double mu, double beta, KeanuRandom random) {
        if (beta <= 0.0) {
            throw new IllegalArgumentException("Invalid value for beta: " + beta);
        }
        if (random.nextDouble() > 0.5) {
            return mu + beta * Math.log(random.nextDouble());
        } else {
            return mu - beta * Math.log(random.nextDouble());
        }
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor muMinusXAbsNegDivBeta = mu.minus(x).abs().divInPlace(beta);
        final DoubleTensor logTwoBeta = beta.times(2.).logInPlace();
        return muMinusXAbsNegDivBeta.plusInPlace(logTwoBeta).unaryMinus();
    }

    public static DoubleVertex logProbOutput(DoublePlaceholderVertex x, DoublePlaceholderVertex mu, DoublePlaceholderVertex beta) {
        final DoubleVertex muMinusXAbsNegDivBeta = mu.minus(x).abs().div(beta);
        final DoubleVertex logTwoBeta = beta.times(2.).log();
        return muMinusXAbsNegDivBeta.plus(logTwoBeta).unaryMinus();
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor muMinusX = mu.minus(x);
        final DoubleTensor muMinusXAbs = muMinusX.abs();

        final DoubleTensor denominator = muMinusXAbs.times(beta);

        final DoubleTensor dLogPdx = muMinusX.divInPlace(denominator);
        final DoubleTensor dLogPdMu = x.minus(mu).divInPlace(denominator);
        final DoubleTensor dLogPdBeta = muMinusXAbs.minusInPlace(beta).divInPlace(beta.pow(2));

        return new Diffs()
            .put(MU, dLogPdMu)
            .put(BETA, dLogPdBeta)
            .put(X, dLogPdx);
    }

}
