package io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.Gamma;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphSupplier;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
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

import static io.improbable.keanu.distributions.hyperparam.Diffs.K;
import static io.improbable.keanu.distributions.hyperparam.Diffs.THETA;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;
import static io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertexWrapper.wrapIfNeeded;

public class GammaVertex extends VertexImpl<DoubleTensor, DoubleVertex> implements DoubleVertex, Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor>, LogProbGraphSupplier {

    private final DoubleVertex theta;
    private final DoubleVertex k;
    private static final String THETA_NAME = "theta";
    private static final String K_NAME = "k";

    /**
     * Theta or k or both driving an arbitrarily shaped tensor of Gamma
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the vertex
     * @param theta       the theta (scale) of the Gamma with either the same shape as specified for this vertex
     * @param k           the k (shape) of the Gamma with either the same shape as specified for this vertex
     */
    public GammaVertex(@LoadShape long[] tensorShape,
                       @LoadVertexParam(THETA_NAME) Vertex<DoubleTensor, ?> theta,
                       @LoadVertexParam(K_NAME) Vertex<DoubleTensor, ?> k) {
        super(TensorShape.getBroadcastResultShape(tensorShape, theta.getShape(), k.getShape()));

        this.theta = wrapIfNeeded(theta);
        this.k = wrapIfNeeded(k);
        setParents(theta, k);
    }

    /**
     * One to one constructor for mapping some shape of theta and k to matching shaped gamma.
     *
     * @param theta the theta (scale) of the Gamma with either the same shape as specified for this vertex
     * @param k     the k (shape) of the Gamma with either the same shape as specified for this vertex
     */
    @ExportVertexToPythonBindings
    public GammaVertex(Vertex<DoubleTensor, ?> theta,
                       Vertex<DoubleTensor, ?> k) {
        this(TensorShape.getBroadcastResultShape(theta.getShape(), k.getShape()), theta, k);
    }

    public GammaVertex(Vertex<DoubleTensor, ?> theta, double k) {
        this(theta, new ConstantDoubleVertex(k));
    }

    public GammaVertex(double theta, Vertex<DoubleTensor, ?> k) {
        this(new ConstantDoubleVertex(theta), k);
    }

    public GammaVertex(double theta, double k) {
        this(new ConstantDoubleVertex(theta), new ConstantDoubleVertex(k));
    }

    @SaveVertexParam(THETA_NAME)
    public DoubleVertex getTheta() {
        return theta;
    }

    @SaveVertexParam(K_NAME)
    public DoubleVertex getK() {
        return k;
    }

    @Override
    public double logProb(DoubleTensor value) {
        DoubleTensor thetaValues = theta.getValue();
        DoubleTensor kValues = k.getValue();

        DoubleTensor logPdfs = Gamma.withParameters(thetaValues, kValues).logProb(value);
        return logPdfs.sumNumber();
    }

    @Override
    public LogProbGraph logProbGraph() {
        final DoublePlaceholderVertex xPlaceholder = new DoublePlaceholderVertex(this.getShape());
        final DoublePlaceholderVertex thetaPlaceholder = new DoublePlaceholderVertex(theta.getShape());
        final DoublePlaceholderVertex kPlaceholder = new DoublePlaceholderVertex(k.getShape());

        return LogProbGraph.builder()
            .input(this, xPlaceholder)
            .input(theta, thetaPlaceholder)
            .input(k, kPlaceholder)
            .logProbOutput(Gamma.logProbOutput(xPlaceholder, thetaPlaceholder, kPlaceholder))
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {
        Diffs dlnP = Gamma.withParameters(theta.getValue(), k.getValue()).dLogProb(value);

        Map<Vertex, DoubleTensor> dLogProbWrtParameters = new HashMap<>();

        if (withRespectTo.contains(theta)) {
            dLogProbWrtParameters.put(theta, dlnP.get(THETA).getValue());
        }

        if (withRespectTo.contains(k)) {
            dLogProbWrtParameters.put(k, dlnP.get(K).getValue());
        }

        if (withRespectTo.contains(this)) {
            dLogProbWrtParameters.put(this, dlnP.get(X).getValue());
        }

        return dLogProbWrtParameters;
    }

    @Override
    public DoubleTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return Gamma.withParameters(theta.getValue(), k.getValue()).sample(shape, random);
    }

}
