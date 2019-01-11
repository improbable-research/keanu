package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.continuous.Beta;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphSupplier;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.distributions.hyperparam.Diffs.A;
import static io.improbable.keanu.distributions.hyperparam.Diffs.B;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;

public class BetaVertex extends DoubleVertex implements Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor>, LogProbGraphSupplier {

    private final DoubleVertex alpha;
    private final DoubleVertex beta;
    private static final String ALPHA_NAME = "alpha";
    private static final String BETA_NAME = "beta";

    /**
     * One alpha or beta or both that match a proposed tensor shape of Beta.
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the tensor contained in the vertex
     * @param alpha       the alpha of the Beta with either the same tensorShape as specified for this vertex or a scalar
     * @param beta        the beta of the Beta with either the same tensorShape as specified for this vertex or a scalar
     */
    public BetaVertex(@LoadShape long[] tensorShape,
                      @LoadVertexParam(ALPHA_NAME) DoubleVertex alpha,
                      @LoadVertexParam(BETA_NAME)DoubleVertex beta) {
        super(tensorShape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(tensorShape, alpha.getShape(), beta.getShape());

        this.alpha = alpha;
        this.beta = beta;
        setParents(alpha, beta);
    }

    ContinuousDistribution distribution() {
        return Beta.withParameters(alpha.getValue(), beta.getValue(), DoubleTensor.scalar(0.), DoubleTensor.scalar(1.));
    }

    /**
     * One to one constructor for mapping some tensorShape of alpha and beta to
     * a matching tensorShaped Beta.
     *
     * @param alpha the alpha of the Beta with either the same tensorShape as specified for this vertex or a scalar
     * @param beta  the beta of the Beta with either the same tensorShape as specified for this vertex or a scalar
     */
    @ExportVertexToPythonBindings
    public BetaVertex(DoubleVertex alpha, DoubleVertex beta) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(alpha.getShape(), beta.getShape()), alpha, beta);
    }

    public BetaVertex(DoubleVertex alpha, double beta) {
        this(alpha, new ConstantDoubleVertex(beta));
    }

    public BetaVertex(double alpha, DoubleVertex beta) {
        this(new ConstantDoubleVertex(alpha), beta);
    }

    public BetaVertex(double alpha, double beta) {
        this(new ConstantDoubleVertex(alpha), new ConstantDoubleVertex(beta));
    }

    public BetaVertex(long[] tensorShape, DoubleVertex alpha, double beta) {
        this(tensorShape, alpha, new ConstantDoubleVertex(beta));
    }

    public BetaVertex(long[] tensorShape, double alpha, DoubleVertex beta) {
        this(tensorShape, new ConstantDoubleVertex(alpha), beta);
    }

    public BetaVertex(long[] tensorShape, double alpha, double beta) {
        this(tensorShape, new ConstantDoubleVertex(alpha), new ConstantDoubleVertex(beta));
    }

    @SaveVertexParam(ALPHA_NAME)
    public DoubleVertex getAlpha() {
        return alpha;
    }

    @SaveVertexParam(BETA_NAME)
    public DoubleVertex getBeta() {
        return beta;
    }

    @Override
    public double logProb(DoubleTensor value) {
        DoubleTensor logPdfs = distribution().logProb(value);
        return logPdfs.sum();
    }

    @Override
    public LogProbGraph logProbGraph() {
        final LogProbGraph.DoublePlaceHolderVertex xPlaceHolder = new LogProbGraph.DoublePlaceHolderVertex(this.getShape());
        final LogProbGraph.DoublePlaceHolderVertex alphaPlaceHolder = new LogProbGraph.DoublePlaceHolderVertex(alpha.getShape());
        final LogProbGraph.DoublePlaceHolderVertex betaPlaceHolder = new LogProbGraph.DoublePlaceHolderVertex(beta.getShape());

        final DoubleVertex lnGammaAlpha = alphaPlaceHolder.logGamma();
        final DoubleVertex lnGammaBeta = betaPlaceHolder.logGamma();
        final DoubleVertex alphaPlusBetaLnGamma = (alphaPlaceHolder.plus(betaPlaceHolder)).logGamma();
        final DoubleVertex alphaMinusOneTimesLnX = xPlaceHolder.log().times(alphaPlaceHolder.minus(1));
        final DoubleVertex betaMinusOneTimesOneMinusXLn = xPlaceHolder.unaryMinus().plus(1).log().times(betaPlaceHolder.minus(1));

        final DoubleVertex betaFunction = lnGammaAlpha.plus(lnGammaBeta).minus(alphaPlusBetaLnGamma);

        final DoubleVertex logProbOutput = alphaMinusOneTimesLnX.plus(betaMinusOneTimesOneMinusXLn).minus(betaFunction);

        return LogProbGraph.builder()
            .input(this, xPlaceHolder)
            .input(alpha, alphaPlaceHolder)
            .input(beta, betaPlaceHolder)
            .logProbOutput(logProbOutput)
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {
        Diffs dlnP = distribution().dLogProb(value);

        Map<Vertex, DoubleTensor> dLogProbWrtParameters = new HashMap<>();

        if (withRespectTo.contains(alpha)) {
            dLogProbWrtParameters.put(alpha, dlnP.get(A).getValue());
        }

        if (withRespectTo.contains(beta)) {
            dLogProbWrtParameters.put(beta, dlnP.get(B).getValue());
        }

        if (withRespectTo.contains(this)) {
            dLogProbWrtParameters.put(this, dlnP.get(X).getValue());
        }

        return dLogProbWrtParameters;
    }

    @Override
    public DoubleTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return distribution().sample(shape, random);
    }

}
