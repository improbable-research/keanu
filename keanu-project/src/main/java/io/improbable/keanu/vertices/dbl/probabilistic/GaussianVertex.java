package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.LogProbAsAGraphable;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.PlaceHolderDoubleVertex;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.distributions.hyperparam.Diffs.MU;
import static io.improbable.keanu.distributions.hyperparam.Diffs.SIGMA;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;

public class GaussianVertex extends DoubleVertex implements Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor>, LogProbAsAGraphable {

    private final DoubleVertex mu;
    private final DoubleVertex sigma;
    protected static final String MU_NAME = "mu";
    protected static final String SIGMA_NAME = "sigma";

    /**
     * One mu or sigma or both that match a proposed tensor shape of Gaussian
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the tensor in this vertex
     * @param mu          the mu of the Gaussian with either the same tensorShape as specified for this vertex or a scalar
     * @param sigma       the sigma of the Gaussian with either the same tensorShape as specified for this vertex or a scalar
     */
    public GaussianVertex(@LoadShape long[] tensorShape,
                          @LoadVertexParam(MU_NAME) DoubleVertex mu,
                          @LoadVertexParam(SIGMA_NAME) DoubleVertex sigma) {
        super(tensorShape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(tensorShape, mu.getShape(), sigma.getShape());

        this.mu = mu;
        this.sigma = sigma;
        setParents(mu, sigma);
    }

    @ExportVertexToPythonBindings
    public GaussianVertex(DoubleVertex mu, DoubleVertex sigma) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(mu.getShape(), sigma.getShape()), mu, sigma);
    }

    public GaussianVertex(DoubleVertex mu, double sigma) {
        this(mu, new ConstantDoubleVertex(sigma));
    }

    public GaussianVertex(double mu, DoubleVertex sigma) {
        this(new ConstantDoubleVertex(mu), sigma);
    }

    public GaussianVertex(double mu, double sigma) {
        this(new ConstantDoubleVertex(mu), new ConstantDoubleVertex(sigma));
    }

    public GaussianVertex(long[] tensorShape, DoubleVertex mu, double sigma) {
        this(tensorShape, mu, new ConstantDoubleVertex(sigma));
    }

    public GaussianVertex(long[] tensorShape, double mu, DoubleVertex sigma) {
        this(tensorShape, new ConstantDoubleVertex(mu), sigma);
    }

    public GaussianVertex(long[] tensorShape, double mu, double sigma) {
        this(tensorShape, new ConstantDoubleVertex(mu), new ConstantDoubleVertex(sigma));
    }

    @SaveVertexParam(MU_NAME)
    public DoubleVertex getMu() {
        return mu;
    }

    @SaveVertexParam(SIGMA_NAME)
    public DoubleVertex getSigma() {
        return sigma;
    }

    @Override
    public double logProb(DoubleTensor value) {

        DoubleTensor muValues = mu.getValue();
        DoubleTensor sigmaValues = sigma.getValue();

        DoubleTensor logPdfs = Gaussian.withParameters(muValues, sigmaValues).logProb(value);

        return logPdfs.sum();
    }

    @Override
    public LogProbGraph logProbGraph() {

        final PlaceHolderDoubleVertex xInput = new PlaceHolderDoubleVertex(this.getShape());
        final PlaceHolderDoubleVertex muInput = new PlaceHolderDoubleVertex(mu.getShape());
        final PlaceHolderDoubleVertex sigmaInput = new PlaceHolderDoubleVertex(sigma.getShape());

        final DoubleVertex lnSigma = sigmaInput.log();
        final DoubleVertex xMinusMuSquared = xInput.minus(muInput).pow(2);
        final DoubleVertex xMinusMuSquaredOver2Variance = xMinusMuSquared.div(sigmaInput.pow(2).times(2.0));

        final DoubleVertex logProbOutput = xMinusMuSquaredOver2Variance.plus(lnSigma).plus(Gaussian.LN_SQRT_2PI).unaryMinus().sum();

        return LogProbGraph.builder()
            .input(this, xInput)
            .input(mu, muInput)
            .input(sigma, sigmaInput)
            .logProbOutput(logProbOutput)
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {
        Diffs dlnP = Gaussian.withParameters(mu.getValue(), sigma.getValue()).dLogProb(value);

        Map<Vertex, DoubleTensor> dLogProbWrtParameters = new HashMap<>();

        if (withRespectTo.contains(mu)) {
            dLogProbWrtParameters.put(mu, dlnP.get(MU).getValue());
        }

        if (withRespectTo.contains(sigma)) {
            dLogProbWrtParameters.put(sigma, dlnP.get(SIGMA).getValue());
        }

        if (withRespectTo.contains(this)) {
            dLogProbWrtParameters.put(this, dlnP.get(X).getValue());
        }

        return dLogProbWrtParameters;
    }

    @Override
    public DoubleTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return Gaussian.withParameters(mu.getValue(), sigma.getValue()).sample(shape, random);
    }

}
