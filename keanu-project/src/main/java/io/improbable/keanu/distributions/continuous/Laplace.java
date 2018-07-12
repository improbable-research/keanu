package io.improbable.keanu.distributions.continuous;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.List;

public class Laplace implements ContinuousDistribution {

    private final DoubleTensor mu;
    private final DoubleTensor beta;

    /**
     * @param mu     location
     * @param beta   shape
     */
    public static ContinuousDistribution withParameters(DoubleTensor mu, DoubleTensor beta) {
        return new Laplace(mu, beta);
    }

    private Laplace(DoubleTensor mu, DoubleTensor beta) {
        this.mu = mu;
        this.beta = beta;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        Tensor.FlattenedView<Double> muWrapped = mu.getFlattenedView();
        Tensor.FlattenedView<Double> betaWrapped = beta.getFlattenedView();

        int length = ArrayUtil.prod(shape);
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
        final DoubleTensor logTwoBeta = beta.times(2).logInPlace();
        return muMinusXAbsNegDivBeta.plusInPlace(logTwoBeta).unaryMinus();
    }

    @Override
    public List<DoubleTensor> dLogProb(DoubleTensor x) {
        final DoubleTensor muMinusX = mu.minus(x);
        final DoubleTensor muMinusXAbs = muMinusX.abs();

        final DoubleTensor denominator = muMinusXAbs.times(beta);

        final DoubleTensor dPdx = muMinusX.divInPlace(denominator);
        final DoubleTensor dPdMu = x.minus(mu).divInPlace(denominator);
        final DoubleTensor dPdBeta = muMinusXAbs.minusInPlace(beta).divInPlace(beta.pow(2));

        return ImmutableList.of(dPdMu, dPdBeta, dPdx);
    }

}
