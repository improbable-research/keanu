package io.improbable.keanu.distributions.continuous;

import static io.improbable.keanu.distributions.dual.Diffs.MU;
import static io.improbable.keanu.distributions.dual.Diffs.SIGMA;
import static io.improbable.keanu.distributions.dual.Diffs.X;

import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.network.KeanuComputationalGraph;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

public class Gaussian {

    public static final double SQRT_2PI = Math.sqrt(Math.PI * 2);
    public static final double LN_SQRT_2PI = Math.log(SQRT_2PI);

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

        logProb.addInput("x", xInput);
        logProb.addInput("mu", muInput);
        logProb.addInput("sigma", sigmaInput);
        logProb.addOutput("logProb", logProbOutput);
    }

    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        DoubleTensor unityGaussian = random.nextGaussian(shape);
        return unityGaussian.timesInPlace(sigma.getValue()).plusInPlace(mu.getValue());
    }

    public DoubleTensor logProb(DoubleTensor x) {

        DoubleTensor muValue = mu.getValue();
        DoubleTensor sigmaValue = sigma.getValue();

        return logProb
            .setInput("x", x)
            .setInput("mu", muValue)
            .setInput("sigma", sigmaValue)
            .calculate("logProb");


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
            .put(MU, dLogPdmu)
            .put(SIGMA, dLogPdsigma)
            .put(X, dLogPdx);
    }

}
