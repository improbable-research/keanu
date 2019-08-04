package io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.continuous.SmoothUniform;
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

import java.util.Collections;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.distributions.hyperparam.Diffs.X;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;
import static io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertexWrapper.wrapIfNeeded;
import static java.util.Collections.singletonMap;

public class SmoothUniformVertex extends VertexImpl<DoubleTensor, DoubleVertex> implements DoubleVertex, Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor>, LogProbGraphSupplier {

    private static final double DEFAULT_EDGE_SHARPNESS = 0.01;

    private final DoubleVertex xMin;
    private final DoubleVertex xMax;
    private final double edgeSharpness;
    private static final String X_MIN_NAME = "xMin";
    private static final String X_MAX_NAME = "xMax";

    public SmoothUniformVertex(@LoadShape long[] tensorShape,
                               @LoadVertexParam(X_MIN_NAME) Vertex<DoubleTensor, ?> xMin,
                               @LoadVertexParam(X_MAX_NAME) Vertex<DoubleTensor, ?> xMax) {
        this(tensorShape, xMin, xMax, DEFAULT_EDGE_SHARPNESS);
    }

    /**
     * One xMin or Xmax or both that match a proposed tensor shape of Smooth Uniform
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape   the desired shape of the vertex
     * @param xMin          the xMin of the Smooth Uniform with either the same shape as specified for this vertex or a scalar
     * @param xMax          the xMax of the Smooth Uniform with either the same shape as specified for this vertex or a scalar
     * @param edgeSharpness the edge sharpness of the Smooth Uniform
     */
    public SmoothUniformVertex(long[] tensorShape, Vertex<DoubleTensor, ?> xMin, Vertex<DoubleTensor, ?> xMax, double edgeSharpness) {
        super(tensorShape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(tensorShape, xMin.getShape(), xMax.getShape());

        this.xMin = wrapIfNeeded(xMin);
        this.xMax = wrapIfNeeded(xMax);
        this.edgeSharpness = edgeSharpness;
        setParents(xMin, xMax);
    }

    /**
     * One to one constructor for mapping some shape of mu and sigma to
     * a matching shaped Smooth Uniform.
     *
     * @param xMin          the xMin of the Smooth Uniform with either the same shape as specified for this vertex or a scalar
     * @param xMax          the xMax of the Smooth Uniform with either the same shape as specified for this vertex or a scalar
     * @param edgeSharpness the edge sharpness of the Smooth Uniform
     */
    public SmoothUniformVertex(Vertex<DoubleTensor, ?> xMin, Vertex<DoubleTensor, ?> xMax, double edgeSharpness) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(xMin.getShape(), xMax.getShape()), xMin, xMax, edgeSharpness);
    }

    public SmoothUniformVertex(Vertex<DoubleTensor, ?> xMin, double xMax, double edgeSharpness) {
        this(xMin, new ConstantDoubleVertex(xMax), edgeSharpness);
    }

    public SmoothUniformVertex(double xMin, Vertex<DoubleTensor, ?> xMax, double edgeSharpness) {
        this(new ConstantDoubleVertex(xMin), xMax, edgeSharpness);
    }

    public SmoothUniformVertex(double xMin, double xMax, double edgeSharpness) {
        this(new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), edgeSharpness);
    }

    @ExportVertexToPythonBindings
    public SmoothUniformVertex(Vertex<DoubleTensor, ?> xMin, Vertex<DoubleTensor, ?> xMax) {
        this(xMin, xMax, DEFAULT_EDGE_SHARPNESS);
    }

    public SmoothUniformVertex(Vertex<DoubleTensor, ?> xMin, double xMax) {
        this(xMin, new ConstantDoubleVertex(xMax), DEFAULT_EDGE_SHARPNESS);
    }

    public SmoothUniformVertex(double xMin, Vertex<DoubleTensor, ?> xMax) {
        this(new ConstantDoubleVertex(xMin), xMax, DEFAULT_EDGE_SHARPNESS);
    }

    public SmoothUniformVertex(double xMin, double xMax) {
        this(new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), DEFAULT_EDGE_SHARPNESS);
    }

    public SmoothUniformVertex(long[] tensorShape, Vertex<DoubleTensor, ?> xMin, double xMax, double edgeSharpness) {
        this(tensorShape, xMin, new ConstantDoubleVertex(xMax), edgeSharpness);
    }

    public SmoothUniformVertex(long[] tensorShape, double xMin, Vertex<DoubleTensor, ?> xMax, double edgeSharpness) {
        this(tensorShape, new ConstantDoubleVertex(xMin), xMax, edgeSharpness);
    }

    public SmoothUniformVertex(long[] tensorShape, double xMin, double xMax, double edgeSharpness) {
        this(tensorShape, new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), edgeSharpness);
    }

    public SmoothUniformVertex(long[] tensorShape, Vertex<DoubleTensor, ?> xMin, double xMax) {
        this(tensorShape, xMin, new ConstantDoubleVertex(xMax), DEFAULT_EDGE_SHARPNESS);
    }

    public SmoothUniformVertex(long[] tensorShape, double xMin, Vertex<DoubleTensor, ?> xMax) {
        this(tensorShape, new ConstantDoubleVertex(xMin), xMax, DEFAULT_EDGE_SHARPNESS);
    }

    public SmoothUniformVertex(long[] tensorShape, double xMin, double xMax) {
        this(tensorShape, new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), DEFAULT_EDGE_SHARPNESS);
    }

    @SaveVertexParam(X_MIN_NAME)
    public DoubleVertex getXMin() {
        return xMin;
    }

    @SaveVertexParam(X_MAX_NAME)
    public DoubleVertex getXMax() {
        return xMax;
    }

    public double getEdgeSharpness() {
        return edgeSharpness;
    }

    @Override
    public double logProb(DoubleTensor value) {
        final DoubleTensor min = xMin.getValue();
        final DoubleTensor max = xMax.getValue();
        final DoubleTensor density = SmoothUniform.withParameters(min, max, this.edgeSharpness).logProb(value);
        return density.sumNumber();
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
            .logProbOutput(SmoothUniform.logProbOutput(xPlaceholder, xMinPlaceholder, xMaxPlaceholder, edgeSharpness))
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {

        if (withRespectTo.contains(this)) {
            final DoubleTensor min = xMin.getValue();
            final DoubleTensor max = xMax.getValue();
            ContinuousDistribution distribution = SmoothUniform.withParameters(min, max, this.edgeSharpness);
            final DoubleTensor dPdx = distribution.dLogProb(value).get(X).getValue();
            final DoubleTensor density = distribution.logProb(value);
            final DoubleTensor dLogPdx = dPdx.divInPlace(density);
            return singletonMap(this, dLogPdx);
        }

        return Collections.emptyMap();
    }

    @Override
    public DoubleTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return SmoothUniform.withParameters(xMin.getValue(), xMax.getValue(), this.edgeSharpness).sample(shape, random);
    }
}
