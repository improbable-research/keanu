package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.Triangular;
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
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;

import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;

public class TriangularVertex extends DoubleVertex implements Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor>, LogProbGraphSupplier {

    private final DoubleVertex xMin;
    private final DoubleVertex xMax;
    private final DoubleVertex c;
    private static final String X_MIN_NAME = "xMin";
    private static final String X_MAX_NAME = "xMax";
    private static final String C_NAME = "c";

    /**
     * One xMin, xMax, c or all three that match a proposed tensor shape of Triangular
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the vertex
     * @param xMin        the xMin of the Triangular with either the same shape as specified for this vertex or a scalar
     * @param xMax        the xMax of the Triangular with either the same shape as specified for this vertex or a scalar
     * @param c           the center of the Triangular with either the same shape as specified for this vertex or a scalar
     */
    public TriangularVertex(@LoadShape long[] tensorShape,
                            @LoadVertexParam(X_MIN_NAME) DoubleVertex xMin,
                            @LoadVertexParam(X_MAX_NAME) DoubleVertex xMax,
                            @LoadVertexParam(C_NAME) DoubleVertex c) {
        super(tensorShape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(tensorShape, xMin.getShape(), xMax.getShape(), c.getShape());

        this.xMin = xMin;
        this.xMax = xMax;
        this.c = c;
        setParents(xMin, xMax, c);
    }

    public TriangularVertex(long[] tensorShape, DoubleVertex xMin, DoubleVertex xMax, double c) {
        this(tensorShape, xMin, xMax, new ConstantDoubleVertex(c));
    }

    public TriangularVertex(long[] tensorShape, DoubleVertex xMin, double xMax, DoubleVertex c) {
        this(tensorShape, xMin, new ConstantDoubleVertex(xMax), c);
    }

    public TriangularVertex(long[] tensorShape, DoubleVertex xMin, double xMax, double c) {
        this(tensorShape, xMin, new ConstantDoubleVertex(xMax), new ConstantDoubleVertex(c));
    }

    public TriangularVertex(long[] tensorShape, double xMin, DoubleVertex xMax, DoubleVertex c) {
        this(tensorShape, new ConstantDoubleVertex(xMin), xMax, c);
    }

    public TriangularVertex(long[] tensorShape, double xMin, double xMax, DoubleVertex c) {
        this(tensorShape, new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), c);
    }

    public TriangularVertex(long[] tensorShape, double xMin, double xMax, double c) {
        this(tensorShape, new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), new ConstantDoubleVertex(c));
    }

    /**
     * One to one constructor for mapping some shape of xMin, xMax and c to a matching shaped triangular.
     *
     * @param xMin the xMin of the Triangular with either the same shape as specified for this vertex or a scalar
     * @param xMax the xMax of the Triangular with either the same shape as specified for this vertex or a scalar
     * @param c    the c of the Triangular with either the same shape as specified for this vertex or a scalar
     */
    @ExportVertexToPythonBindings
    public TriangularVertex(DoubleVertex xMin, DoubleVertex xMax, DoubleVertex c) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(xMin.getShape(), xMax.getShape(), c.getShape()), xMin, xMax, c);
    }

    public TriangularVertex(DoubleVertex xMin, DoubleVertex xMax, double c) {
        this(xMin, xMax, new ConstantDoubleVertex(c));
    }

    public TriangularVertex(DoubleVertex xMin, double xMax, DoubleVertex c) {
        this(xMin, new ConstantDoubleVertex(xMax), c);
    }

    public TriangularVertex(DoubleVertex xMin, double xMax, double c) {
        this(xMin, new ConstantDoubleVertex(xMax), new ConstantDoubleVertex(c));
    }

    public TriangularVertex(double xMin, DoubleVertex xMax, DoubleVertex c) {
        this(new ConstantDoubleVertex(xMin), xMax, c);
    }

    public TriangularVertex(double xMin, double xMax, DoubleVertex c) {
        this(new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), c);
    }

    public TriangularVertex(double xMin, double xMax, double c) {
        this(new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), new ConstantDoubleVertex(c));
    }

    @SaveVertexParam(X_MIN_NAME)
    public DoubleVertex getXMin() {
        return xMin;
    }

    @SaveVertexParam(X_MAX_NAME)
    public DoubleVertex getXMax() {
        return xMax;
    }

    @SaveVertexParam(C_NAME)
    public DoubleVertex getC() {
        return c;
    }

    @Override
    public double logProb(DoubleTensor value) {
        DoubleTensor xMinValues = xMin.getValue();
        DoubleTensor xMaxValues = xMax.getValue();
        DoubleTensor cValues = c.getValue();

        DoubleTensor logPdfs = Triangular.withParameters(xMinValues, xMaxValues, cValues).logProb(value);
        return logPdfs.sum();
    }

    @Override
    public LogProbGraph logProbGraph() {
        final LogProbGraph.DoublePlaceholderVertex xPlaceholder = new LogProbGraph.DoublePlaceholderVertex(this.getShape());
        final LogProbGraph.DoublePlaceholderVertex xMinPlaceholder = new LogProbGraph.DoublePlaceholderVertex(xMin.getShape());
        final LogProbGraph.DoublePlaceholderVertex xMaxPlaceholder = new LogProbGraph.DoublePlaceholderVertex(xMax.getShape());
        final LogProbGraph.DoublePlaceholderVertex cPlaceholder = new LogProbGraph.DoublePlaceholderVertex(c.getShape());

        final DoubleVertex range = xMaxPlaceholder.minus(xMinPlaceholder);

        final BoolVertex conditionalFirstHalf = xPlaceholder.greaterThan(xMinPlaceholder);
        final BoolVertex conditionalSecondHalf = xPlaceholder.lessThan(cPlaceholder);
        final BoolVertex conditionalAnd = conditionalFirstHalf.and(conditionalSecondHalf);
        final DoubleVertex conditionalResult = If
            .isTrue(conditionalAnd)
            .then(ConstantVertex.of(1.).div(range).times(2).times(xPlaceholder.minus(xMinPlaceholder)).div(cPlaceholder.minus(xMinPlaceholder)))
            .orElse(new ConstantDoubleVertex(DoubleTensor.zeros(conditionalAnd.getShape())));

        final BoolVertex elseIfConditionalFirstHalf = xPlaceholder.greaterThan(c);
        final BoolVertex elseIfConditionalSecondHalf = xPlaceholder.lessThan(xMax);
        final BoolVertex elseIfConditionalAnd = elseIfConditionalFirstHalf.and(elseIfConditionalSecondHalf);
        final DoubleVertex elseIfConditionalResult = If
            .isTrue(elseIfConditionalAnd)
            .then(ConstantVertex.of(2.).div(range).times(xMaxPlaceholder.minus(xPlaceholder)).div(xMaxPlaceholder.minus(cPlaceholder)))
            .orElse(new ConstantDoubleVertex(DoubleTensor.zeros(elseIfConditionalAnd.getShape())));

        final DoubleVertex logProbOutput = conditionalResult.plus(elseIfConditionalResult.plus(elseIfConditionalResult)).log();

        return LogProbGraph.builder()
            .input(this, xPlaceholder)
            .input(xMin, xMinPlaceholder)
            .input(xMax, xMaxPlaceholder)
            .input(c, cPlaceholder)
            .logProbOutput(logProbOutput)
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DoubleTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return Triangular.withParameters(xMin.getValue(), xMax.getValue(), c.getValue()).sample(shape, random);
    }
}
