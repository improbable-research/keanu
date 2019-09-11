package io.improbable.keanu.vertices.tensor.bool.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.discrete.Bernoulli;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Samplable;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphSupplier;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.bool.BooleanPlaceholderVertex;
import io.improbable.keanu.vertices.tensor.bool.BooleanVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;
import static io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertexWrapper.wrapIfNeeded;

public class BernoulliVertex extends VertexImpl<BooleanTensor, BooleanVertex> implements BooleanVertex, ProbabilisticBoolean, Samplable<BooleanTensor>, LogProbGraphSupplier {

    private final DoubleVertex probTrue;
    private final static String PROBTRUE_NAME = "probTrue";

    /**
     * One probTrue that must match a proposed tensor shape of Bernoulli.
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param shape    the desired shape of the vertex
     * @param probTrue the probability the bernoulli returns true
     */
    public BernoulliVertex(@LoadShape long[] shape,
                           @LoadVertexParam(PROBTRUE_NAME) Vertex<DoubleTensor, ?> probTrue) {
        super(shape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(shape, probTrue.getShape());
        this.probTrue = wrapIfNeeded(probTrue);
        setParents(probTrue);
    }

    /**
     * One to one constructor for mapping some shape of probTrue to
     * a matching shaped Bernoulli.
     *
     * @param probTrue probTrue with same shape as desired Bernoulli tensor or scalar
     */
    @ExportVertexToPythonBindings
    public BernoulliVertex(Vertex<DoubleTensor, ?> probTrue) {
        this(probTrue.getShape(), probTrue);
    }

    public BernoulliVertex(double probTrue) {
        this(Tensor.SCALAR_SHAPE, new ConstantDoubleVertex(probTrue));
    }

    public BernoulliVertex(long[] shape, double probTrue) {
        this(shape, new ConstantDoubleVertex(probTrue));
    }

    @SaveVertexParam(PROBTRUE_NAME)
    public DoubleVertex getProbTrue() {
        return probTrue;
    }

    @Override
    public double logProb(BooleanTensor value) {
        return Bernoulli.withParameters(probTrue.getValue()).logProb(value).sumNumber();
    }

    @Override
    public LogProbGraph logProbGraph() {
        BooleanPlaceholderVertex valuePlaceholder = new BooleanPlaceholderVertex(this.getShape());
        DoublePlaceholderVertex probTruePlaceholder = new DoublePlaceholderVertex(probTrue.getShape());

        return LogProbGraph.builder()
            .input(this, valuePlaceholder)
            .input(probTrue, probTruePlaceholder)
            .logProbOutput(Bernoulli.logProbGraph(valuePlaceholder, probTruePlaceholder))
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(BooleanTensor value, Set<? extends Vertex> withRespectTo) {

        if (!(probTrue.isDifferentiable())) {
            throw new UnsupportedOperationException("The probability of the Bernoulli being true must be differentiable");
        }

        if (withRespectTo.contains(probTrue)) {
            DoubleTensor dLogPdp = Bernoulli.withParameters(probTrue.getValue()).dLogProb(value);
            return Collections.singletonMap(probTrue, dLogPdp);
        }

        return Collections.emptyMap();
    }

    @Override
    public BooleanTensor sample(long[] shape, KeanuRandom random) {
        return Bernoulli.withParameters(probTrue.getValue()).sample(shape, random);
    }
}
