package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.nd4j.linalg.util.ArrayUtil;

public class Laplace {

    private Laplace() {
    }

    /**
     * @param shape  shape of tensor returned
     * @param mu     location
     * @param beta   shape
     * @param random source of randomness
     * @return a random number from the Laplace distribution
     */
    public static DoubleTensor sample(int[] shape, DoubleTensor mu, DoubleTensor beta, KeanuRandom random) {
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

    public static DoubleTensor logPdf(DoubleTensor mu, DoubleTensor beta, DoubleTensor x) {
        final DoubleTensor muMinusXAbsNegDivBeta = mu.minus(x).abs().divInPlace(beta);
        final DoubleTensor logTwoBeta = beta.times(2).logInPlace();
        return muMinusXAbsNegDivBeta.plusInPlace(logTwoBeta).unaryMinus();
    }

    public static Diff dlnPdf(DoubleTensor mu, DoubleTensor beta, DoubleTensor x) {
        final DoubleTensor muMinusX = mu.minus(x);
        final DoubleTensor muMinusXAbs = muMinusX.abs();

        final DoubleTensor denominator = muMinusXAbs.times(beta);

        final DoubleTensor dPdx = muMinusX.divInPlace(denominator);
        final DoubleTensor dPdMu = x.minus(mu).divInPlace(denominator);
        final DoubleTensor dPdBeta = muMinusXAbs.minusInPlace(beta).divInPlace(beta.pow(2));

        return new Diff(dPdMu, dPdBeta, dPdx);
    }

    public static class Diff {
        public final DoubleTensor dPdmu;
        public final DoubleTensor dPdbeta;
        public final DoubleTensor dPdx;

        public Diff(DoubleTensor dPdmu, DoubleTensor dPdbeta, DoubleTensor dPdx) {
            this.dPdmu = dPdmu;
            this.dPdbeta = dPdbeta;
            this.dPdx = dPdx;
        }
    }

}
