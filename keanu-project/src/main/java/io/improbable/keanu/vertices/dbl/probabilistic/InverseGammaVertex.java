package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.InverseGamma;
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

public class InverseGammaVertex extends DoubleVertex implements Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor>, LogProbGraphSupplier {

    private final DoubleVertex alpha;
    private final DoubleVertex beta;
    private static final String ALPHA_NAME = "alpha";
    private static final String BETA_NAME = "beta";

    /**
     * One alpha or beta or both driving an arbitrarily shaped tensor of Inverse Gamma
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the vertex
     * @param alpha       the alpha of the Inverse Gamma with either the same shape as specified for this vertex or alpha scalar
     * @param beta        the beta of the Inverse Gamma with either the same shape as specified for this vertex or alpha scalar
     */
    public InverseGammaVertex(@LoadShape long[] tensorShape,
                              @LoadVertexParam(ALPHA_NAME) DoubleVertex alpha,
                              @LoadVertexParam(BETA_NAME) DoubleVertex beta) {
        super(tensorShape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(tensorShape, alpha.getShape(), beta.getShape());

        this.alpha = alpha;
        this.beta = beta;
        setParents(alpha, beta);
    }

    /**
     * One to one constructor for mapping some shape of alpha and beta to
     * alpha matching shaped Inverse Gamma.
     *
     * @param alpha the alpha of the Inverse Gamma with either the same shape as specified for this vertex or alpha scalar
     * @param beta  the beta of the Inverse Gamma with either the same shape as specified for this vertex or alpha scalar
     */
    @ExportVertexToPythonBindings
    public InverseGammaVertex(DoubleVertex alpha, DoubleVertex beta) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(alpha.getShape(), beta.getShape()), alpha, beta);
    }

    public InverseGammaVertex(DoubleVertex alpha, double beta) {
        this(alpha, new ConstantDoubleVertex(beta));
    }

    public InverseGammaVertex(double alpha, DoubleVertex beta) {
        this(new ConstantDoubleVertex(alpha), beta);
    }

    public InverseGammaVertex(double alpha, double beta) {
        this(new ConstantDoubleVertex(alpha), new ConstantDoubleVertex(beta));
    }

    public InverseGammaVertex(long[] tensorShape, DoubleVertex alpha, double beta) {
        this(tensorShape, alpha, new ConstantDoubleVertex(beta));
    }

    public InverseGammaVertex(long[] tensorShape, double alpha, DoubleVertex beta) {
        this(tensorShape, new ConstantDoubleVertex(alpha), beta);
    }

    public InverseGammaVertex(long[] tensorShape, double alpha, double beta) {
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
        DoubleTensor alphaValues = alpha.getValue();
        DoubleTensor betaValues = beta.getValue();

        DoubleTensor logPdfs = InverseGamma.withParameters(alphaValues, betaValues).logProb(value);
        return logPdfs.sum();
    }

    @Override
    public LogProbGraph logProbGraph() {
        final LogProbGraph.DoublePlaceholderVertex xPlaceholder = new LogProbGraph.DoublePlaceholderVertex(this.getShape());
        final LogProbGraph.DoublePlaceholderVertex alphaPlaceholder = new LogProbGraph.DoublePlaceholderVertex(alpha.getShape());
        final LogProbGraph.DoublePlaceholderVertex betaPlaceholder = new LogProbGraph.DoublePlaceholderVertex(beta.getShape());

        final DoubleVertex aTimesLnB = alphaPlaceholder.times(betaPlaceholder.log());
        final DoubleVertex negAMinus1TimesLnX = xPlaceholder.log().times(alphaPlaceholder.unaryMinus().minus(1));
        final DoubleVertex lnGammaA = alphaPlaceholder.logGamma();

        final DoubleVertex logProbOutput = aTimesLnB.plus(negAMinus1TimesLnX).minus(lnGammaA).minus(betaPlaceholder.div(xPlaceholder));

        return LogProbGraph.builder()
            .input(this, xPlaceholder)
            .input(alpha, alphaPlaceholder)
            .input(beta, betaPlaceholder)
            .logProbOutput(logProbOutput)
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {
        Diffs dlnP = InverseGamma.withParameters(alpha.getValue(), beta.getValue()).dLogProb(value);

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
        return InverseGamma.withParameters(alpha.getValue(), beta.getValue()).sample(shape, random);
    }

}
