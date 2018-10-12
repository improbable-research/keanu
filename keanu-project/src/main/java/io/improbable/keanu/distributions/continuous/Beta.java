package io.improbable.keanu.distributions.continuous;

import static io.improbable.keanu.distributions.hyperparam.Diffs.A;
import static io.improbable.keanu.distributions.hyperparam.Diffs.B;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.PlaceHolderDoubleVertex;

public class Beta implements ContinuousDistribution {

    private final DoubleVertex x;
    private final DoubleVertex alpha;
    private final DoubleVertex beta;
    private final DoubleVertex xMin;
    private final DoubleVertex xMax;

    private final LogProbGraph logProbGraph;

    public static Beta withParameters(DoubleVertex x, DoubleVertex alpha, DoubleVertex beta, DoubleVertex xMin, DoubleVertex xMax) {
        return new Beta(x, alpha, beta, xMin, xMax);
    }

    private Beta(DoubleVertex x, DoubleVertex alpha, DoubleVertex beta, DoubleVertex xMin, DoubleVertex xMax) {
        this.x = x;
        this.alpha = alpha;
        this.beta = beta;
        this.xMin = xMin;
        this.xMax = xMax;
        this.logProbGraph = logProbGraph();
    }

    @Override
    public DoubleTensor sample(long[] shape, KeanuRandom random) {

        final DoubleTensor xMaxValue = xMax.getValue();
        final DoubleTensor xMinValue = xMin.getValue();
        final DoubleTensor alphaValue = alpha.getValue();
        final DoubleTensor betaValue = beta.getValue();

        final DoubleTensor y1 = random.nextGamma(shape, DoubleTensor.ONE_SCALAR, alphaValue);
        final DoubleTensor y2 = random.nextGamma(shape, DoubleTensor.ONE_SCALAR, betaValue);

        final DoubleTensor range = xMaxValue.minus(xMinValue);
        final DoubleTensor y1PlusY2 = y1.plus(y2);

        final DoubleTensor lessThan = xMaxValue.minus(y2.div(y1PlusY2).timesInPlace(range));
        final DoubleTensor greaterThan = xMinValue.plus(y1.div(y1PlusY2).timesInPlace(range));

        final DoubleTensor lessMask = alphaValue.getLessThanMask(betaValue);
        final DoubleTensor greaterMask = alphaValue.getGreaterThanOrEqualToMask(betaValue);

        return lessMask.timesInPlace(lessThan).plusInPlace(greaterMask.timesInPlace(greaterThan));
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        logProbGraph.getInput(this.x).setValue(x);
        logProbGraph.getInput(this.alpha).setValue(alpha.getValue());
        logProbGraph.getInput(this.beta).setValue(beta.getValue());
        return logProbGraph.getLogProbOutput().eval();
    }

    public LogProbGraph logProbGraph() {

        final PlaceHolderDoubleVertex xInput = new PlaceHolderDoubleVertex();
        final PlaceHolderDoubleVertex alphaInput = new PlaceHolderDoubleVertex();
        final PlaceHolderDoubleVertex betaInput = new PlaceHolderDoubleVertex();

        final DoubleVertex lnGammaAlpha = alphaInput.logGamma();
        final DoubleVertex lnGammaBeta = betaInput.logGamma();
        final DoubleVertex alphaPlusBetaLnGamma = alphaInput.plus(betaInput).logGamma();
        final DoubleVertex alphaMinusOneTimesLnX = xInput.log().times(alphaInput.minus(1));
        final DoubleVertex betaMinusOneTimesOneMinusXLn = xInput.unaryMinus().plus(1).log().times(betaInput.minus(1));

        final DoubleVertex betaFunction = lnGammaAlpha.plus(lnGammaBeta).minus(alphaPlusBetaLnGamma);

        final DoubleVertex logProbOutput = alphaMinusOneTimesLnX.plus(betaMinusOneTimesOneMinusXLn).minus(betaFunction).sum();

        return new LogProbGraph(logProbOutput)
            .addInput(x, xInput)
            .addInput(alpha, alphaInput)
            .addInput(beta, betaInput);
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor alphaValue = alpha.getValue();
        final DoubleTensor betaValue = beta.getValue();

        final DoubleTensor oneMinusX = x.unaryMinus().plusInPlace(1);
        final DoubleTensor digammaAlphaPlusBeta = alphaValue.plus(betaValue).digammaInPlace();
        final DoubleTensor alphaMinusOneDivX = x.reciprocal().timesInPlace(alphaValue.minus(1));

        final DoubleTensor dLogPdx = alphaMinusOneDivX.minusInPlace(oneMinusX.reciprocal().timesInPlace(betaValue.minus(1)));
        final DoubleTensor dLogPda = x.log().plusInPlace(digammaAlphaPlusBeta.minus(alphaValue.digamma()));
        final DoubleTensor dLogPdb = oneMinusX.logInPlace().plusInPlace(digammaAlphaPlusBeta.minusInPlace(betaValue.digamma()));

        return new Diffs()
            .put(A, dLogPda)
            .put(B, dLogPdb)
            .put(X, dLogPdx);
    }
}
