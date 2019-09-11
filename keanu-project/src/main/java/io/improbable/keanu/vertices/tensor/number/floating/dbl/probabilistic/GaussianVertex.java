package io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
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
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.tensor.TensorShape.getBroadcastResultShape;
import static io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertexWrapper.wrapIfNeeded;

public class GaussianVertex extends VertexImpl<DoubleTensor, DoubleVertex>
    implements ProbabilisticDouble, Differentiable, LogProbGraphSupplier {

    private final DoubleVertex mu;
    private final DoubleVertex sigma;
    protected static final String MU_NAME = "mu";
    protected static final String SIGMA_NAME = "sigma";

    /**
     * A single variate gaussian distribution with mu and sigma. Alternatively, providing multiple mu and sigmas can
     * be used to create a diagonal covariance multivariate gaussian distribution. If a full covariance matrix is
     * needed then use the MultivariateGaussianVertex.
     *
     * @param shape a shape that is broadcastable with the shape of mu and sigma. This shape can be used for batching
     *              but most commonly will match the broadcasted shape of mu with sigma.
     * @param mu    the mu of the Gaussian with a shape that is broadcastable with the shape parameter and the shape
     *              of sigma.
     * @param sigma the sigma of the Gaussian with a shape that is broadcastable with the shape parameter and the shape
     *              of mu.
     */
    public GaussianVertex(@LoadShape long[] shape,
                          @LoadVertexParam(MU_NAME) Vertex<DoubleTensor, ?> mu,
                          @LoadVertexParam(SIGMA_NAME) Vertex<DoubleTensor, ?> sigma) {
        super(getBroadcastResultShape(shape, mu.getShape(), sigma.getShape()));

        this.mu = wrapIfNeeded(mu);
        this.sigma = wrapIfNeeded(sigma);
        setParents(mu, sigma);
    }

    @ExportVertexToPythonBindings
    public GaussianVertex(Vertex<DoubleTensor, ?> mu, Vertex<DoubleTensor, ?> sigma) {
        this(getBroadcastResultShape(mu.getShape(), sigma.getShape()), mu, sigma);
    }

    public GaussianVertex(Vertex<DoubleTensor, ?> mu, double sigma) {
        this(mu, new ConstantDoubleVertex(sigma));
    }

    public GaussianVertex(double mu, Vertex<DoubleTensor, ?> sigma) {
        this(new ConstantDoubleVertex(mu), sigma);
    }

    public GaussianVertex(double mu, double sigma) {
        this(new ConstantDoubleVertex(mu), new ConstantDoubleVertex(sigma));
    }

    public GaussianVertex(long[] tensorShape, Vertex<DoubleTensor, ?> mu, double sigma) {
        this(tensorShape, mu, new ConstantDoubleVertex(sigma));
    }

    public GaussianVertex(long[] tensorShape, double mu, Vertex<DoubleTensor, ?> sigma) {
        this(tensorShape, new ConstantDoubleVertex(mu), sigma);
    }

    public GaussianVertex(long[] tensorShape, double mu, double sigma) {
        this(tensorShape, new ConstantDoubleVertex(mu), new ConstantDoubleVertex(sigma));
    }

    public GaussianVertex(long[] tensorShape, DoubleTensor mu, DoubleTensor sigma) {
        this(tensorShape, new ConstantDoubleVertex(mu), new ConstantDoubleVertex(sigma));
    }

    public GaussianVertex(DoubleTensor mu, DoubleTensor sigma) {
        this(getBroadcastResultShape(mu.getShape(), sigma.getShape()), new ConstantDoubleVertex(mu), new ConstantDoubleVertex(sigma));
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
            .logProbOutput(Gaussian.logProbOutput(xPlaceholder, muPlaceholder, sigmaPlaceholder).sum())
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {

        final boolean wrtX = withRespectTo.contains(this);
        final boolean wrtMu = withRespectTo.contains(mu);
        final boolean wrtSigma = withRespectTo.contains(sigma);

        final DoubleTensor[] dlnP = Gaussian.withParameters(mu.getValue(), sigma.getValue()).dLogProb(value, wrtX, wrtMu, wrtSigma);

        final Map<Vertex, DoubleTensor> dLogProbWrtParameters = new HashMap<>();

        if (withRespectTo.contains(this)) {
            dLogProbWrtParameters.put(this, dlnP[0]);
        }

        if (wrtMu) {
            dLogProbWrtParameters.put(mu, dlnP[1]);
        }

        if (withRespectTo.contains(sigma)) {
            dLogProbWrtParameters.put(sigma, dlnP[2]);
        }

        return dLogProbWrtParameters;
    }

    @Override
    public DoubleTensor sample(long[] shape, KeanuRandom random) {
        return Gaussian.withParameters(mu.getValue(), sigma.getValue()).sample(shape, random);
    }

}
