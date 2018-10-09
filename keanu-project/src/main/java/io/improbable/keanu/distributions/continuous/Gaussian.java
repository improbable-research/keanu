package io.improbable.keanu.distributions.continuous;


import static io.improbable.keanu.distributions.hyperparam.Diffs.MU;
import static io.improbable.keanu.distributions.hyperparam.Diffs.SIGMA;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;

import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.PlaceHolderDoubleVertex;

public class Gaussian {

    public static final double SQRT_2PI = Math.sqrt(Math.PI * 2);
    public static final double LN_SQRT_2PI = Math.log(SQRT_2PI);

    private final DoubleVertex x;
    private final DoubleVertex mu;
    private final DoubleVertex sigma;

    public static Gaussian withParameters(DoubleTensor mu, DoubleTensor sigma) {
        return new Gaussian(new ConstantDoubleVertex(0), new ConstantDoubleVertex(mu), new ConstantDoubleVertex(sigma));
    }

    public static Gaussian withParameters(DoubleVertex x, DoubleVertex mu, DoubleVertex sigma) {
        return new Gaussian(x, mu, sigma);
    }

    private Gaussian(DoubleVertex x, DoubleVertex mu, DoubleVertex sigma) {
        this.x = x;
        this.mu = mu;
        this.sigma = sigma;
    }

    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        DoubleTensor unityGaussian = random.nextGaussian(shape);
        return unityGaussian.timesInPlace(sigma.getValue()).plusInPlace(mu.getValue());
    }

    public DoubleTensor logProb(DoubleTensor xValue) {

        final DoubleTensor muValue = mu.getValue();
        final DoubleTensor sigmaValue = sigma.getValue();

        final DoubleTensor lnSigma = sigmaValue.log();
        final DoubleTensor xMinusMuSquared = xValue.minus(muValue).powInPlace(2);
        final DoubleTensor xMinusMuSquaredOver2Variance = xMinusMuSquared.divInPlace(sigmaValue.pow(2).timesInPlace(2.0));
        return xMinusMuSquaredOver2Variance.plusInPlace(lnSigma).plusInPlace(LN_SQRT_2PI).unaryMinusInPlace();
    }

    public LogProbGraph logProbGraph() {

        final PlaceHolderDoubleVertex xInput = new PlaceHolderDoubleVertex();
        final PlaceHolderDoubleVertex muInput = new PlaceHolderDoubleVertex();
        final PlaceHolderDoubleVertex sigmaInput = new PlaceHolderDoubleVertex();

        final DoubleVertex lnSigma = sigmaInput.log();
        final DoubleVertex xMinusMuSquared = xInput.minus(muInput).pow(2);
        final DoubleVertex xMinusMuSquaredOver2Variance = xMinusMuSquared.div(sigmaInput.pow(2).times(2.0));
        final DoubleVertex logProbOutput = xMinusMuSquaredOver2Variance.plus(lnSigma).plus(Gaussian.LN_SQRT_2PI).unaryMinus().sum();

        return new LogProbGraph(logProbOutput)
            .addInput(x, xInput)
            .addInput(mu, muInput)
            .addInput(sigma, sigmaInput);
    }

    public Diffs dLogProb(DoubleTensor xValue) {

        final DoubleTensor muValue = mu.getValue();
        final DoubleTensor sigmaValue = sigma.getValue();

        final DoubleTensor variance = sigmaValue.pow(2);
        final DoubleTensor xMinusMu = xValue.minus(muValue);

        final DoubleTensor dLogPdmu = xMinusMu.div(variance);
        final DoubleTensor dLogPdx = dLogPdmu.unaryMinus();
        final DoubleTensor dLogPdsigma = xMinusMu.powInPlace(2)
            .divInPlace(variance.timesInPlace(sigmaValue))
            .minusInPlace(sigmaValue.reciprocal());

        return new Diffs()
            .put(MU, dLogPdmu)
            .put(SIGMA, dLogPdsigma)
            .put(X, dLogPdx);
    }

}
