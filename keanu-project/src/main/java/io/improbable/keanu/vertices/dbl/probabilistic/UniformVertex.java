package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.Uniform;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonSingularShapeOrAllSingular;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;
import static java.util.Collections.singletonMap;

public class UniformVertex extends DoubleVertex implements Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor> {

    private final DoubleVertex xMin;
    private final DoubleVertex xMax;

    /**
     * One xMin or xMax or both that match a proposed tensor shape of Uniform Vertex
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape desired tensor shape
     * @param xMin        the inclusive lower bound of the Uniform with either the same shape as specified for this vertex or a scalar
     * @param xMax        the exclusive upper bound of the Uniform with either the same shape as specified for this vertex or a scalar
     */
    public UniformVertex(long[] tensorShape, DoubleVertex xMin, DoubleVertex xMax) {
        super(tensorShape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(tensorShape, xMin.getShape(), xMax.getShape());

        this.xMin = xMin;
        this.xMax = xMax;
        setParents(xMin, xMax);
    }

    /**
     * One to one constructor for mapping some shape of mu and sigma to
     * a matching shaped Uniform Vertex
     *
     * @param xMin the inclusive lower bound of the Uniform with either the same shape as specified for this vertex or a scalar
     * @param xMax the exclusive upper bound of the Uniform with either the same shape as specified for this vertex or a scalar
     */
    @ExportVertexToPythonBindings
    public UniformVertex(DoubleVertex xMin, DoubleVertex xMax) {
        this(checkHasOneNonSingularShapeOrAllSingular(xMin.getShape(), xMax.getShape()), xMin, xMax);
    }

    public UniformVertex(DoubleVertex xMin, double xMax) {
        this(xMin, new ConstantDoubleVertex(xMax));
    }

    public UniformVertex(double xMin, DoubleVertex xMax) {
        this(new ConstantDoubleVertex(xMin), xMax);
    }

    public UniformVertex(double xMin, double xMax) {
        this(new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax));
    }

    public UniformVertex(long[] tensorShape, DoubleVertex xMin, double xMax) {
        this(tensorShape, xMin, new ConstantDoubleVertex(xMax));
    }

    public UniformVertex(long[] tensorShape, double xMin, DoubleVertex xMax) {
        this(tensorShape, new ConstantDoubleVertex(xMin), xMax);
    }

    public UniformVertex(long[] tensorShape, double xMin, double xMax) {
        this(tensorShape, new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax));
    }

    public DoubleVertex getXMin() {
        return xMin;
    }

    public DoubleVertex getXMax() {
        return xMax;
    }

    @Override
    public double logProb(DoubleTensor value) {
        return Uniform.withParameters(xMin.getValue(), xMax.getValue()).logProb(value).sum();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {

        if (withRespectTo.contains(this)) {
            DoubleTensor dLogPdx = DoubleTensor.zeros(this.xMax.getShape());
            dLogPdx = dLogPdx.setWithMaskInPlace(value.getGreaterThanMask(xMax.getValue()), Double.NEGATIVE_INFINITY);
            dLogPdx = dLogPdx.setWithMaskInPlace(value.getLessThanOrEqualToMask(xMin.getValue()), Double.POSITIVE_INFINITY);

            return singletonMap(this, dLogPdx);
        }

        return Collections.emptyMap();
    }

    @Override
    public DoubleTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return Uniform.withParameters(xMin.getValue(), xMax.getValue()).sample(shape, random);
    }

}
