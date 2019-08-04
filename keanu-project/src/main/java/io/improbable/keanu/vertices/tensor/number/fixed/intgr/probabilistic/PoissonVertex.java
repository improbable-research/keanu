package io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.discrete.Poisson;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphSupplier;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerPlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;
import static io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertexWrapper.wrapIfNeeded;

public class PoissonVertex extends VertexImpl<IntegerTensor, IntegerVertex> implements IntegerVertex, ProbabilisticInteger, SamplableWithManyScalars<IntegerTensor>, LogProbGraphSupplier {

    private final DoubleVertex mu;
    private static final String MU_NAME = "mu";

    /**
     * One mu that must match a proposed tensor shape of Poisson.
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param shape the desired shape of the vertex
     * @param mu    the mu of the Poisson with either the same shape as specified for this vertex or a scalar
     */
    public PoissonVertex(@LoadShape long[] shape,
                         @LoadVertexParam(MU_NAME) Vertex<DoubleTensor, ?> mu) {
        super(shape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(shape, mu.getShape());

        this.mu = wrapIfNeeded(mu);
        setParents(mu);
    }

    public PoissonVertex(long[] shape, double mu) {
        this(shape, new ConstantDoubleVertex(mu));
    }

    /**
     * One to one constructor for mapping some shape of mu to
     * a matching shaped Poisson.
     *
     * @param mu mu with same shape as desired Poisson tensor or scalar
     */
    @ExportVertexToPythonBindings
    public PoissonVertex(Vertex<DoubleTensor, ?> mu) {
        this(mu.getShape(), mu);
    }

    public PoissonVertex(double mu) {
        this(Tensor.SCALAR_SHAPE, new ConstantDoubleVertex(mu));
    }

    @SaveVertexParam(MU_NAME)
    public DoubleVertex getMu() {
        return mu;
    }

    @Override
    public double logProb(IntegerTensor value) {
        return Poisson.withParameters(mu.getValue()).logProb(value).sumNumber();
    }

    @Override
    public LogProbGraph logProbGraph() {
        IntegerPlaceholderVertex valuePlaceholder = new IntegerPlaceholderVertex(this.getShape());
        DoublePlaceholderVertex muPlaceholder = new DoublePlaceholderVertex(mu.getShape());

        return LogProbGraph.builder()
            .input(this, valuePlaceholder)
            .input(mu, muPlaceholder)
            .logProbOutput(Poisson.logProbOutput(valuePlaceholder, muPlaceholder))
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(IntegerTensor value, Set<? extends Vertex> withRespectTo) {
        return Collections.emptyMap();
    }

    @Override
    public IntegerTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return Poisson.withParameters(mu.getValue()).sample(shape, random);
    }
}
