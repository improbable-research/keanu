package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.discrete.UniformInt;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
import io.improbable.keanu.vertices.SaveParentVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;

import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class UniformIntVertex extends IntegerVertex implements ProbabilisticInteger, SamplableWithManyScalars<IntegerTensor> {

    private IntegerVertex min;
    private IntegerVertex max;
    private static final String MIN_NAME = "min";
    private static final String MAX_NAME = "max";

    /**
     * @param shape tensor shape of value
     * @param min   The inclusive lower bound.
     * @param max   The exclusive upper bound.
     */
    public UniformIntVertex(long[] shape, IntegerVertex min, IntegerVertex max) {
        super(shape);
        checkTensorsMatchNonScalarShapeOrAreScalar(shape, min.getShape(), max.getShape());

        this.min = min;
        this.max = max;
        setParents(min, max);
    }

    public UniformIntVertex(long[] shape, int min, int max) {
        this(shape, new ConstantIntegerVertex(min), new ConstantIntegerVertex(max));
    }

    public UniformIntVertex(long[] shape, IntegerTensor min, IntegerTensor max) {
        this(shape, new ConstantIntegerVertex(min), new ConstantIntegerVertex(max));
    }

    public UniformIntVertex(long[] shape, IntegerVertex min, int max) {
        this(shape, min, new ConstantIntegerVertex(max));
    }

    public UniformIntVertex(long[] shape, int min, IntegerVertex max) {
        this(shape, new ConstantIntegerVertex(min), max);
    }

    @ExportVertexToPythonBindings
    public UniformIntVertex(@LoadParentVertex(MIN_NAME) IntegerVertex min, @LoadParentVertex(MAX_NAME) IntegerVertex max) {
        this(checkHasSingleNonScalarShapeOrAllScalar(min.getShape(), max.getShape()), min, max);
    }

    public UniformIntVertex(IntegerVertex min, int max) {
        this(min.getShape(), min, new ConstantIntegerVertex(max));
    }

    public UniformIntVertex(int min, IntegerVertex max) {
        this(max.getShape(), new ConstantIntegerVertex(min), max);
    }

    public UniformIntVertex(int min, int max) {
        this(Tensor.SCALAR_SHAPE, new ConstantIntegerVertex(min), new ConstantIntegerVertex(max));
    }

    @SaveParentVertex(MIN_NAME)
    public IntegerVertex getMin() {
        return min;
    }

    @SaveParentVertex(MAX_NAME)
    public IntegerVertex getMax() {
        return max;
    }

    @Override
    public double logProb(IntegerTensor value) {
        return UniformInt.withParameters(min.getValue(), max.getValue()).logProb(value).sum();
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
