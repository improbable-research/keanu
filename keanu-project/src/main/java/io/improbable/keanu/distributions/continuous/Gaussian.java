package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.network.KeanuComputationalGraph;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

public class Gaussian {

    public static final double SQRT_2PI = Math.sqrt(Math.PI * 2);
    public static final double LN_SQRT_2PI = Math.log(SQRT_2PI);

    private static final String X = "x";
    private static final String MU = "mu";
    private static final String SIGMA = "sigma";
    private static final String LOG_PROB = "logProb";

    private final DoubleVertex mu;
    private final DoubleVertex sigma;
    private final KeanuComputationalGraph logProb;

    public static Gaussian withParameters(DoubleVertex mu, DoubleVertex sigma) {
        return new Gaussian(mu, sigma);
    }

    private Gaussian(DoubleVertex mu, DoubleVertex sigma) {
        this.mu = mu;
        this.sigma = sigma;

        logProb = new KeanuComputationalGraph();

        ConstantDoubleVertex xInput = new ConstantDoubleVertex(0);
        ConstantDoubleVertex muInput = new ConstantDoubleVertex(0);
        ConstantDoubleVertex sigmaInput = new ConstantDoubleVertex(0);
        DoubleVertex logProbOutput = logProbGraph(xInput, muInput, sigmaInput);

        logProb.addInput(X, xInput);
        logProb.addInput(MU, muInput);
        logProb.addInput(SIGMA, sigmaInput);
        logProb.addOutput(LOG_PROB, logProbOutput);
    }

    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        DoubleTensor unityGaussian = random.nextGaussian(shape);
        return unityGaussian.timesInPlace(sigma.getValue()).plusInPlace(mu.getValue());
    }

    public DoubleTensor logProb(DoubleTensor x) {

        DoubleTensor muValue = mu.getValue();
        DoubleTensor sigmaValue = sigma.getValue();

        return logProb
            .setInput(X, x)
            .setInput(MU, muValue)
            .setInput(SIGMA, sigmaValue)
            .calculate(LOG_PROB);


//        final DoubleTensor lnSigma = sigmaValue.log();
//        final DoubleTensor xMinusMuSquared = x.minus(muValue).powInPlace(2);
//        final DoubleTensor xMinusMuSquaredOver2Variance = xMinusMuSquared.divInPlace(sigmaValue.pow(2).timesInPlace(2.0));
//        return xMinusMuSquaredOver2Variance.plusInPlace(lnSigma).plusInPlace(LN_SQRT_2PI).unaryMinusInPlace();
    }

    public static DoubleVertex logProbGraph(DoubleVertex x, DoubleVertex mu, DoubleVertex sigma) {

        final DoubleVertex lnSigma = sigma.log();
        final DoubleVertex xMinusMuSquared = x.minus(mu).pow(2);
        final DoubleVertex xMinusMuSquaredOver2Variance = xMinusMuSquared.div(sigma.pow(2).times(2.0));

        return xMinusMuSquaredOver2Variance.plus(lnSigma).plus(Gaussian.LN_SQRT_2PI).unaryMinus();
    }

    public Diffs dLogProb(DoubleTensor x) {
        DoubleTensor muValue = mu.getValue();
        DoubleTensor sigmaValue = sigma.getValue();

        final DoubleTensor variance = sigmaValue.pow(2);
        final DoubleTensor xMinusMu = x.minus(muValue);

        final DoubleTensor dLogPdmu = xMinusMu.div(variance);
        final DoubleTensor dLogPdx = dLogPdmu.unaryMinus();
        final DoubleTensor dLogPdsigma = xMinusMu.powInPlace(2)
            .divInPlace(variance.timesInPlace(sigmaValue))
            .minusInPlace(sigmaValue.reciprocal());

        return new Diffs()
            .put(Diffs.MU, dLogPdmu)
            .put(Diffs.SIGMA, dLogPdsigma)
            .put(Diffs.X, dLogPdx);
    }

}
