package io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.Triangular;
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

import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;
import static io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertexWrapper.wrapIfNeeded;

public class TriangularVertex extends VertexImpl<DoubleTensor, DoubleVertex> implements DoubleVertex, Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor>, LogProbGraphSupplier {

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
                            @LoadVertexParam(X_MIN_NAME) Vertex<DoubleTensor, ?> xMin,
                            @LoadVertexParam(X_MAX_NAME) Vertex<DoubleTensor, ?> xMax,
                            @LoadVertexParam(C_NAME) Vertex<DoubleTensor, ?> c) {
        super(tensorShape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(tensorShape, xMin.getShape(), xMax.getShape(), c.getShape());

        this.xMin = wrapIfNeeded(xMin);
        this.xMax = wrapIfNeeded(xMax);
        this.c = wrapIfNeeded(c);
        setParents(xMin, xMax, c);
    }

    public TriangularVertex(long[] tensorShape, Vertex<DoubleTensor, ?> xMin, Vertex<DoubleTensor, ?> xMax, double c) {
        this(tensorShape, xMin, xMax, new ConstantDoubleVertex(c));
    }

    public TriangularVertex(long[] tensorShape, Vertex<DoubleTensor, ?> xMin, double xMax, Vertex<DoubleTensor, ?> c) {
        this(tensorShape, xMin, new ConstantDoubleVertex(xMax), c);
    }

    public TriangularVertex(long[] tensorShape, Vertex<DoubleTensor, ?> xMin, double xMax, double c) {
        this(tensorShape, xMin, new ConstantDoubleVertex(xMax), new ConstantDoubleVertex(c));
    }

    public TriangularVertex(long[] tensorShape, double xMin, Vertex<DoubleTensor, ?> xMax, Vertex<DoubleTensor, ?> c) {
        this(tensorShape, new ConstantDoubleVertex(xMin), xMax, c);
    }

    public TriangularVertex(long[] tensorShape, double xMin, double xMax, Vertex<DoubleTensor, ?> c) {
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
    public TriangularVertex(Vertex<DoubleTensor, ?> xMin, Vertex<DoubleTensor, ?> xMax, Vertex<DoubleTensor, ?> c) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(xMin.getShape(), xMax.getShape(), c.getShape()), xMin, xMax, c);
    }

    public TriangularVertex(Vertex<DoubleTensor, ?> xMin, Vertex<DoubleTensor, ?> xMax, double c) {
        this(xMin, xMax, new ConstantDoubleVertex(c));
    }

    public TriangularVertex(Vertex<DoubleTensor, ?> xMin, double xMax, Vertex<DoubleTensor, ?> c) {
        this(xMin, new ConstantDoubleVertex(xMax), c);
    }

    public TriangularVertex(Vertex<DoubleTensor, ?> xMin, double xMax, double c) {
        this(xMin, new ConstantDoubleVertex(xMax), new ConstantDoubleVertex(c));
    }

    public TriangularVertex(double xMin, Vertex<DoubleTensor, ?> xMax, Vertex<DoubleTensor, ?> c) {
        this(new ConstantDoubleVertex(xMin), xMax, c);
    }

    public TriangularVertex(double xMin, double xMax, Vertex<DoubleTensor, ?> c) {
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
        return logPdfs.sumNumber();
    }

    @Override
    public LogProbGraph logProbGraph() {
        final DoublePlaceholderVertex xPlaceholder = new DoublePlaceholderVertex(this.getShape());
        final DoublePlaceholderVertex xMinPlaceholder = new DoublePlaceholderVertex(xMin.getShape());
        final DoublePlaceholderVertex xMaxPlaceholder = new DoublePlaceholderVertex(xMax.getShape());
        final DoublePlaceholderVertex cPlaceholder = new DoublePlaceholderVertex(c.getShape());

        return LogProbGraph.builder()
            .input(this, xPlaceholder)
            .input(xMin, xMinPlaceholder)
            .input(xMax, xMaxPlaceholder)
            .input(c, cPlaceholder)
            .logProbOutput(Triangular.logProbOutput(xPlaceholder, xMinPlaceholder, xMaxPlaceholder, cPlaceholder))
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

    @Override
    public DoubleTensor upperBound() {
        return xMax.getValue();
    }

    @Override
    public DoubleTensor lowerBound() {
        return xMin.getValue();
    }
}
