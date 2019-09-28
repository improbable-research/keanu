package io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.LogNormal;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphSupplier;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.distributions.hyperparam.Diffs.MU;
import static io.improbable.keanu.distributions.hyperparam.Diffs.SIGMA;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;
import static io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertexWrapper.wrapIfNeeded;

public class LogNormalVertex extends VertexImpl<DoubleTensor, DoubleVertex>
    implements ProbabilisticDouble, Differentiable, LogProbGraphSupplier {

    private final DoubleVertex mu;
    private final DoubleVertex sigma;
    private static final String MU_NAME = "mu";
    private static final String SIGMA_NAME = "sigma";

    /**
     * One mu or s or both driving an arbitrarily shaped tensor of LogNormal
     * https://en.wikipedia.org/wiki/Log-normal_distribution
     *
     * @param tensorShape the desired shape of the vertex
     * @param mu          the mu (location) of the LogNormal with either the same tensor shape as specified for this
     *                    vertex or mu scalar
     * @param sigma       the sigma of the Logistic with either the same shape as specified for this vertex or mu scalar
     */
    public LogNormalVertex(@LoadShape long[] tensorShape,
                           @LoadVertexParam(MU_NAME) Vertex<DoubleTensor, ?> mu,
                           @LoadVertexParam(SIGMA_NAME) Vertex<DoubleTensor, ?> sigma) {
        super(TensorShape.getBroadcastResultShape(tensorShape, mu.getShape(), sigma.getShape()));

        this.mu = wrapIfNeeded(mu);
        this.sigma = wrapIfNeeded(sigma);
        setParents(mu, sigma);
    }

    public LogNormalVertex(long[] tensorShape, Vertex<DoubleTensor, ?> mu, double sigma) {
        this(tensorShape, mu, ConstantVertex.of(sigma));
    }

    public LogNormalVertex(long[] tensorShape, double mu, Vertex<DoubleTensor, ?> sigma) {
        this(tensorShape, ConstantVertex.of(mu), sigma);
    }

    public LogNormalVertex(long[] tensorShape, double mu, double sigma) {
        this(tensorShape, ConstantVertex.of(mu), ConstantVertex.of(sigma));
    }

    @ExportVertexToPythonBindings
    public LogNormalVertex(Vertex<DoubleTensor, ?> mu, Vertex<DoubleTensor, ?> sigma) {
        this(TensorShape.getBroadcastResultShape(mu.getShape(), sigma.getShape()), mu, sigma);
    }

    public LogNormalVertex(double mu, Vertex<DoubleTensor, ?> sigma) {
        this(ConstantVertex.of(mu), sigma);
    }

    public LogNormalVertex(Vertex<DoubleTensor, ?> mu, double sigma) {
        this(mu, ConstantVertex.of(sigma));
    }

    public LogNormalVertex(double mu, double sigma) {
        this(ConstantVertex.of(mu), ConstantVertex.of(sigma));
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

        DoubleTensor logPdfs = LogNormal.withParameters(muValues, sigmaValues).logProb(value);

        return logPdfs.sumNumber();
    }

    @Override
    public LogProbGraph logProbGraph() {
        final DoublePlaceholderVertex xPlaceholder = new DoublePlaceholderVertex(this.getShape());
        final DoublePlaceholderVertex muPlaceholder = new DoublePlaceholderVertex(mu.getShape());
        final DoublePlaceholderVertex sigmaPlaceholder = new DoublePlaceholderVertex(sigma.getShape());

        return LogProbGraph.builder()
            .input(this, xPlaceholder)
            .input(mu, muPlaceholder)
            .input(sigma, sigmaPlaceholder)
            .output(LogNormal.logProbOutput(xPlaceholder, muPlaceholder, sigmaPlaceholder))
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {
        Diffs dlnP = LogNormal.withParameters(mu.getValue(), sigma.getValue()).dLogProb(value);

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
    public DoubleTensor sample(long[] shape, KeanuRandom random) {
        return LogNormal.withParameters(mu.getValue(), sigma.getValue()).sample(shape, random);
    }

    @Override
    public DoubleTensor lowerBound() {
        return DoubleTensor.scalar(0.0);
    }
}
