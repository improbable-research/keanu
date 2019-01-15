package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.continuous.SmoothUniform;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphSupplier;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.distributions.hyperparam.Diffs.X;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import static java.util.Collections.singletonMap;

public class SmoothUniformVertex extends DoubleVertex implements Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor>, LogProbGraphSupplier {

    private static final double DEFAULT_EDGE_SHARPNESS = 0.01;

    private final DoubleVertex xMin;
    private final DoubleVertex xMax;
    private final double edgeSharpness;
    private static final String X_MIN_NAME = "xMin";
    private static final String X_MAX_NAME = "xMax";

    public SmoothUniformVertex(@LoadShape long[] tensorShape,
                               @LoadVertexParam(X_MIN_NAME) DoubleVertex xMin,
                               @LoadVertexParam(X_MAX_NAME) DoubleVertex xMax) {
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
    public SmoothUniformVertex(long[] tensorShape, DoubleVertex xMin, DoubleVertex xMax, double edgeSharpness) {
        super(tensorShape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(tensorShape, xMin.getShape(), xMax.getShape());

        this.xMin = xMin;
        this.xMax = xMax;
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
    public SmoothUniformVertex(DoubleVertex xMin, DoubleVertex xMax, double edgeSharpness) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(xMin.getShape(), xMax.getShape()), xMin, xMax, edgeSharpness);
    }

    public SmoothUniformVertex(DoubleVertex xMin, double xMax, double edgeSharpness) {
        this(xMin, new ConstantDoubleVertex(xMax), edgeSharpness);
    }

    public SmoothUniformVertex(double xMin, DoubleVertex xMax, double edgeSharpness) {
        this(new ConstantDoubleVertex(xMin), xMax, edgeSharpness);
    }

    public SmoothUniformVertex(double xMin, double xMax, double edgeSharpness) {
        this(new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), edgeSharpness);
    }

    @ExportVertexToPythonBindings
    public SmoothUniformVertex(DoubleVertex xMin, DoubleVertex xMax) {
        this(xMin, xMax, DEFAULT_EDGE_SHARPNESS);
    }

    public SmoothUniformVertex(DoubleVertex xMin, double xMax) {
        this(xMin, new ConstantDoubleVertex(xMax), DEFAULT_EDGE_SHARPNESS);
    }

    public SmoothUniformVertex(double xMin, DoubleVertex xMax) {
        this(new ConstantDoubleVertex(xMin), xMax, DEFAULT_EDGE_SHARPNESS);
    }

    public SmoothUniformVertex(double xMin, double xMax) {
        this(new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), DEFAULT_EDGE_SHARPNESS);
    }

    public SmoothUniformVertex(long[] tensorShape, DoubleVertex xMin, double xMax, double edgeSharpness) {
        this(tensorShape, xMin, new ConstantDoubleVertex(xMax), edgeSharpness);
    }

    public SmoothUniformVertex(long[] tensorShape, double xMin, DoubleVertex xMax, double edgeSharpness) {
        this(tensorShape, new ConstantDoubleVertex(xMin), xMax, edgeSharpness);
    }

    public SmoothUniformVertex(long[] tensorShape, double xMin, double xMax, double edgeSharpness) {
        this(tensorShape, new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), edgeSharpness);
    }

    public SmoothUniformVertex(long[] tensorShape, DoubleVertex xMin, double xMax) {
        this(tensorShape, xMin, new ConstantDoubleVertex(xMax), DEFAULT_EDGE_SHARPNESS);
    }

    public SmoothUniformVertex(long[] tensorShape, double xMin, DoubleVertex xMax) {
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
        return density.sum();
    }

    @Override
    public LogProbGraph logProbGraph() {
        final LogProbGraph.DoublePlaceholderVertex xPlaceholder = new LogProbGraph.DoublePlaceholderVertex(this.getShape());
        final LogProbGraph.DoublePlaceholderVertex xMinPlaceholder = new LogProbGraph.DoublePlaceholderVertex(xMin.getShape());
        final LogProbGraph.DoublePlaceholderVertex xMaxPlaceholder = new LogProbGraph.DoublePlaceholderVertex(xMax.getShape());

        final DoubleVertex bodyWidth = xMaxPlaceholder.minus(xMinPlaceholder);
        final DoubleVertex shoulderWidth = bodyWidth.times(edgeSharpness);
        final DoubleVertex rightCutoff = xMaxPlaceholder.plus(shoulderWidth);
        final DoubleVertex leftCutoff = xMinPlaceholder.minus(shoulderWidth);

        BoolVertex firstConditional = xPlaceholder.greaterThanOrEqualTo(xMinPlaceholder)
            .and(xPlaceholder.lessThanOrEqualTo(xMaxPlaceholder));
        final DoubleVertex firstConditionalResult = If
            .isTrue(firstConditional)
            .then(bodyHeight(shoulderWidth, bodyWidth))
            .orElse(new ConstantDoubleVertex(DoubleTensor.zeros(firstConditional.getShape())));

        BoolVertex secondConditional = xPlaceholder.lessThan(xMinPlaceholder)
            .and(xPlaceholder.greaterThan(leftCutoff));
        final DoubleVertex secondConditionalResult = If
            .isTrue(secondConditional)
            .then(shoulder(shoulderWidth, bodyWidth, xPlaceholder.minus(leftCutoff)))
            .orElse(new ConstantDoubleVertex(DoubleTensor.zeros(firstConditional.getShape())));

        BoolVertex thirdConditional = xPlaceholder.greaterThan(xMaxPlaceholder)
            .and(xPlaceholder.lessThan(rightCutoff));
        final DoubleVertex thirdConditionalResult = If
            .isTrue(thirdConditional)
            .then(shoulder(shoulderWidth, bodyWidth, shoulderWidth.minus(xPlaceholder).plus(xMaxPlaceholder)))
            .orElse(new ConstantDoubleVertex(DoubleTensor.zeros(firstConditional.getShape())));

        final DoubleVertex logProbOutput = firstConditionalResult
            .plus(secondConditionalResult)
            .plus(thirdConditionalResult)
            .log();

        return LogProbGraph.builder()
            .input(this, xPlaceholder)
            .input(xMin, xMinPlaceholder)
            .input(xMax, xMaxPlaceholder)
            .logProbOutput(logProbOutput)
            .build();
    }

    private DoubleVertex bodyHeight(DoubleVertex shoulderWidth, DoubleVertex bodyWidth) {
        return ConstantVertex.of(1.).div(shoulderWidth.plus(bodyWidth));
    }

    private static DoubleVertex shoulder(DoubleVertex Sw, DoubleVertex Bw, DoubleVertex x) {
        final DoubleVertex A = getCubeCoefficient(Sw, Bw);
        final DoubleVertex B = getSquareCoefficient(Sw, Bw);
        return x.pow(3).times(A).plus(x.pow(2).times(B));
    }

    private static DoubleVertex getCubeCoefficient(DoubleVertex Sw, DoubleVertex Bw) {
        return ConstantVertex.of(-2.).div(Sw.pow(3).times(Sw.plus(Bw)));
    }

    private static DoubleVertex getSquareCoefficient(DoubleVertex Sw, DoubleVertex Bw) {
        return ConstantVertex.of(3.).div(Sw.pow(2).times(Sw.plus(Bw)));
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
