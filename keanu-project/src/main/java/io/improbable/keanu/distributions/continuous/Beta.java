package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.PlaceHolderDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import static io.improbable.keanu.distributions.hyperparam.Diffs.A;
import static io.improbable.keanu.distributions.hyperparam.Diffs.B;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;

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

    public LogProbGraph logProbGraph() {

        final PlaceHolderDoubleVertex xInput = new PlaceHolderDoubleVertex(x.getShape());
        final PlaceHolderDoubleVertex alphaInput = new PlaceHolderDoubleVertex(alpha.getShape());
        final PlaceHolderDoubleVertex betaInput = new PlaceHolderDoubleVertex(beta.getShape());

        final DoubleVertex lnGammaAlpha = alphaInput.logGamma();
        final DoubleVertex lnGammaBeta = betaInput.logGamma();
        final DoubleVertex alphaPlusBetaLnGamma = alphaInput.plus(betaInput).logGamma();
        final DoubleVertex alphaMinusOneTimesLnX = xInput.log().times(alphaInput.minus(1));
        final DoubleVertex betaMinusOneTimesOneMinusXLn = xInput.unaryMinus().plus(1).log().times(betaInput.minus(1));

        final DoubleVertex betaFunction = lnGammaAlpha.plus(lnGammaBeta).minus(alphaPlusBetaLnGamma);

        final DoubleVertex logProbOutput = alphaMinusOneTimesLnX.plus(betaMinusOneTimesOneMinusXLn).minus(betaFunction).sum();

        return LogProbGraph.builder()
            .input(x, xInput)
            .input(alpha, alphaInput)
            .input(beta, betaInput)
            .logProbOutput(logProbOutput)
            .build();
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        logProbGraph.getInput(this.x).setValue(x);
        logProbGraph.getInput(this.alpha).setValue(alpha.getValue());
        logProbGraph.getInput(this.beta).setValue(beta.getValue());
        return logProbGraph.getLogProbOutput().eval();
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {

        Vertex<DoubleTensor> placeHolderX = logProbGraph.getInput(this.x);
        Vertex<DoubleTensor> placeHolderAlpha = logProbGraph.getInput(this.alpha);
        Vertex<DoubleTensor> placeHolderBeta = logProbGraph.getInput(this.beta);

        placeHolderX.setValue(x);
        placeHolderAlpha.setValue(alpha.getValue());
        placeHolderBeta.setValue(beta.getValue());

        PartialDerivatives dLogProb = Differentiator.reverseModeAutoDiff(
            logProbGraph.getLogProbOutput(),
            (DoubleVertex) placeHolderAlpha,
            (DoubleVertex) placeHolderBeta,
            (DoubleVertex) placeHolderX
        );

        return new Diffs()
            .put(A, dLogProb.withRespectTo(placeHolderAlpha))
            .put(B, dLogProb.withRespectTo(placeHolderBeta))
            .put(X, dLogProb.withRespectTo(placeHolderX));
    }
}
