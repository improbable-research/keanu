package io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.Exponential;
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

import static io.improbable.keanu.distributions.hyperparam.Diffs.LAMBDA;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;
import static io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertexWrapper.wrapIfNeeded;

public class ExponentialVertex extends VertexImpl<DoubleTensor, DoubleVertex> implements DoubleVertex, Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor>, LogProbGraphSupplier {

    private final DoubleVertex rate;
    private static final String RATE_NAME = "rate";

    /**
     * Lambda driving an arbitrarily shaped tensor of Exponential
     * <p>
     * pdf = rate * exp(-rate*x)
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the vertex
     * @param rate        the rate of the Exponential with either be the same shape as specified for this
     *                    vertex or scalar.
     */
    public ExponentialVertex(@LoadShape long[] tensorShape,
                             @LoadVertexParam(RATE_NAME) Vertex<DoubleTensor, ?> rate) {
        super(TensorShape.getBroadcastResultShape(tensorShape, rate.getShape()));

        this.rate = wrapIfNeeded(rate);
        setParents(rate);
    }

    /**
     * One to one constructor for mapping some shape of rate to matching shaped exponential.
     *
     * @param rate the rate of the Exponential with either the same shape as specified for this vertex or scalar
     */
    @ExportVertexToPythonBindings
    public ExponentialVertex(Vertex<DoubleTensor, ?> rate) {
        this(rate.getShape(), rate);
    }

    public ExponentialVertex(double rate) {
        this(new ConstantDoubleVertex(rate));
    }

    @SaveVertexParam(RATE_NAME)
    public DoubleVertex getRate() {
        return rate;
    }

    @Override
    public double logProb(DoubleTensor value) {

        DoubleTensor lambdaValues = rate.getValue();

        DoubleTensor logPdfs = Exponential.withParameters(lambdaValues).logProb(value);

        return logPdfs.sumNumber();
    }

    @Override
    public LogProbGraph logProbGraph() {
        final DoublePlaceholderVertex xPlaceholder = new DoublePlaceholderVertex(this.getShape());
        final DoublePlaceholderVertex ratePlaceholder = new DoublePlaceholderVertex(rate.getShape());

        return LogProbGraph.builder()
            .input(this, xPlaceholder)
            .input(rate, ratePlaceholder)
            .logProbOutput(Exponential.logProbOutput(xPlaceholder, ratePlaceholder))
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {
        Diffs dlnP = Exponential.withParameters(rate.getValue()).dLogProb(value);

        Map<Vertex, DoubleTensor> dLogProbWrtParameters = new HashMap<>();

        if (withRespectTo.contains(rate)) {
            dLogProbWrtParameters.put(rate, dlnP.get(LAMBDA).getValue());
        }

        if (withRespectTo.contains(this)) {
            dLogProbWrtParameters.put(this, dlnP.get(X).getValue());
        }

        return dLogProbWrtParameters;
    }

    @Override
    public DoubleTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return Exponential.withParameters(rate.getValue()).sample(shape, random);
    }

}
