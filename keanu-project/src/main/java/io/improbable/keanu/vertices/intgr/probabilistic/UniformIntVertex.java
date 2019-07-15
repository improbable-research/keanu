package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.discrete.UniformInt;
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
import io.improbable.keanu.vertices.intgr.IntegerPlaceholderVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;

import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;
import static io.improbable.keanu.vertices.intgr.IntegerVertexWrapper.wrapIfNeeded;

public class UniformIntVertex extends VertexImpl<IntegerTensor, IntegerVertex> implements IntegerVertex, ProbabilisticInteger, SamplableWithManyScalars<IntegerTensor>, LogProbGraphSupplier {

    private IntegerVertex min;
    private IntegerVertex max;
    private static final String MIN_NAME = "min";
    private static final String MAX_NAME = "max";

    /**
     * @param shape tensor shape of value
     * @param min   The inclusive lower bound.
     * @param max   The exclusive upper bound.
     */
    public UniformIntVertex(@LoadShape long[] shape,
                            @LoadVertexParam(MIN_NAME) Vertex<IntegerTensor, ?> min,
                            @LoadVertexParam(MAX_NAME) Vertex<IntegerTensor, ?> max) {
        super(shape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(shape, min.getShape(), max.getShape());

        this.min = wrapIfNeeded(min);
        this.max = wrapIfNeeded(max);
        setParents(min, max);
    }

    public UniformIntVertex(long[] shape, int min, int max) {
        this(shape, new ConstantIntegerVertex(min), new ConstantIntegerVertex(max));
    }

    public UniformIntVertex(long[] shape, IntegerTensor min, IntegerTensor max) {
        this(shape, new ConstantIntegerVertex(min), new ConstantIntegerVertex(max));
    }

    public UniformIntVertex(long[] shape, Vertex<IntegerTensor, ?> min, int max) {
        this(shape, min, new ConstantIntegerVertex(max));
    }

    public UniformIntVertex(long[] shape, int min, Vertex<IntegerTensor, ?> max) {
        this(shape, new ConstantIntegerVertex(min), max);
    }

    @ExportVertexToPythonBindings
    public UniformIntVertex(Vertex<IntegerTensor, ?> min, Vertex<IntegerTensor, ?> max) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(min.getShape(), max.getShape()), min, max);
    }

    public UniformIntVertex(Vertex<IntegerTensor, ?> min, int max) {
        this(min.getShape(), min, new ConstantIntegerVertex(max));
    }

    public UniformIntVertex(int min, Vertex<IntegerTensor, ?> max) {
        this(max.getShape(), new ConstantIntegerVertex(min), max);
    }

    public UniformIntVertex(int min, int max) {
        this(Tensor.SCALAR_SHAPE, new ConstantIntegerVertex(min), new ConstantIntegerVertex(max));
    }

    @SaveVertexParam(MIN_NAME)
    public IntegerVertex getMin() {
        return min;
    }

    @SaveVertexParam(MAX_NAME)
    public IntegerVertex getMax() {
        return max;
    }

    @Override
    public double logProb(IntegerTensor value) {
        return UniformInt.withParameters(min.getValue(), max.getValue()).logProb(value).sumNumber();
    }

    @Override
    public LogProbGraph logProbGraph() {
        IntegerPlaceholderVertex valuePlaceholder = new IntegerPlaceholderVertex(this.getShape());
        IntegerPlaceholderVertex minPlaceholder = new IntegerPlaceholderVertex(min.getShape());
        IntegerPlaceholderVertex maxPlaceholder = new IntegerPlaceholderVertex(max.getShape());

        return LogProbGraph.builder()
            .input(this, valuePlaceholder)
            .input(min, minPlaceholder)
            .input(max, maxPlaceholder)
            .logProbOutput(UniformInt.logProbOutput(valuePlaceholder, minPlaceholder, maxPlaceholder))
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(IntegerTensor value, Set<? extends Vertex> withRespectTo) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IntegerTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return UniformInt.withParameters(min.getValue(), max.getValue()).sample(shape, random);
    }
}
