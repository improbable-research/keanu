package io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.Uniform;
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

import java.util.Collections;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;
import static io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertexWrapper.wrapIfNeeded;
import static java.util.Collections.singletonMap;

public class UniformVertex extends VertexImpl<DoubleTensor, DoubleVertex>
    implements ProbabilisticDouble, Differentiable, LogProbGraphSupplier {

    private final DoubleVertex xMin;
    private final DoubleVertex xMax;
    private static final String X_MIN_NAME = "xMin";
    private static final String X_MAX_NAME = "xMax";

    /**
     * One xMin or xMax or both that match a proposed tensor shape of Uniform Vertex
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape desired tensor shape
     * @param xMin        the inclusive lower bound of the Uniform with either the same shape as specified for this vertex or a scalar
     * @param xMax        the exclusive upper bound of the Uniform with either the same shape as specified for this vertex or a scalar
     */
    public UniformVertex(@LoadShape long[] tensorShape,
                         @LoadVertexParam(X_MIN_NAME) Vertex<DoubleTensor, ?> xMin,
                         @LoadVertexParam(X_MAX_NAME) Vertex<DoubleTensor, ?> xMax) {
        super(tensorShape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(tensorShape, xMin.getShape(), xMax.getShape());

        this.xMin = wrapIfNeeded(xMin);
        this.xMax = wrapIfNeeded(xMax);
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
    public UniformVertex(Vertex<DoubleTensor, ?> xMin, Vertex<DoubleTensor, ?> xMax) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(xMin.getShape(), xMax.getShape()), xMin, xMax);
    }

    public UniformVertex(Vertex<DoubleTensor, ?> xMin, double xMax) {
        this(xMin, new ConstantDoubleVertex(xMax));
    }

    public UniformVertex(double xMin, Vertex<DoubleTensor, ?> xMax) {
        this(new ConstantDoubleVertex(xMin), xMax);
    }

    public UniformVertex(double xMin, double xMax) {
        this(new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax));
    }

    public UniformVertex(long[] tensorShape, Vertex<DoubleTensor, ?> xMin, double xMax) {
        this(tensorShape, xMin, new ConstantDoubleVertex(xMax));
    }

    public UniformVertex(long[] tensorShape, double xMin, Vertex<DoubleTensor, ?> xMax) {
        this(tensorShape, new ConstantDoubleVertex(xMin), xMax);
    }

    public UniformVertex(long[] tensorShape, double xMin, double xMax) {
        this(tensorShape, new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax));
    }

    @SaveVertexParam(X_MIN_NAME)
    public DoubleVertex getXMin() {
        return xMin;
    }

    @SaveVertexParam(X_MAX_NAME)
    public DoubleVertex getXMax() {
        return xMax;
    }

    @Override
    public double logProb(DoubleTensor value) {
        return Uniform.withParameters(xMin.getValue(), xMax.getValue()).logProb(value).sumNumber();
    }

    @Override
    public LogProbGraph logProbGraph() {
        final DoublePlaceholderVertex xPlaceholder = new DoublePlaceholderVertex(this.getShape());
        final DoublePlaceholderVertex xMinPlaceholder = new DoublePlaceholderVertex(xMin.getShape());
        final DoublePlaceholderVertex xMaxPlaceholder = new DoublePlaceholderVertex(xMax.getShape());

        return LogProbGraph.builder()
            .input(this, xPlaceholder)
            .input(xMin, xMinPlaceholder)
            .input(xMax, xMaxPlaceholder)
            .logProbOutput(Uniform.logProbOutput(xPlaceholder, xMinPlaceholder, xMaxPlaceholder))
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {

        if (withRespectTo.contains(this)) {
            DoubleTensor dLogPdx = DoubleTensor.zeros(this.xMax.getShape());
            dLogPdx = dLogPdx.setWithMaskInPlace(value.greaterThanMask(xMax.getValue()), 0.0);
            dLogPdx = dLogPdx.setWithMaskInPlace(value.lessThanOrEqualToMask(xMin.getValue()), 0.0);

            return singletonMap(this, dLogPdx);
        }

        return Collections.emptyMap();
    }

    @Override
    public DoubleTensor sample(long[] shape, KeanuRandom random) {
        return Uniform.withParameters(xMin.getValue(), xMax.getValue()).sample(shape, random);
    }

    @Override
    public DoubleTensor upperBound() {
        return xMax.getValue();
    }

    @Override
    public DoubleTensor lowerBound() {
        return xMin.getValue();
    }

}
