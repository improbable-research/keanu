package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Triangular;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class TriangularVertex extends ProbabilisticDouble {

    private final DoubleVertex xMin;
    private final DoubleVertex xMax;
    private final DoubleVertex c;

    /**
     * One xMin, xMax, c or all three driving an arbitrarily shaped tensor of Triangular
     *
     * @param tensorShape the desired shape of the vertex
     * @param xMin        the a of the Triangular with either the same shape as specified for this vertex or a scalar
     * @param xMax        the b of the Triangular with either the same shape as specified for this vertex or a scalar
     * @param c           the center of the Triangular with either the same shape as specified for this vertex or a scalar
     */
    public TriangularVertex(int[] tensorShape, DoubleVertex xMin, DoubleVertex xMax, DoubleVertex c) {

        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, xMin.getShape(), xMax.getShape(), c.getShape());

        this.xMin = xMin;
        this.xMax = xMax;
        this.c = c;
        setParents(xMin, xMax, c);
        setValue(DoubleTensor.placeHolder(tensorShape));
    }

    public TriangularVertex(int[] tensorShape, DoubleVertex xMin, DoubleVertex xMax, double c) {
        this(tensorShape, xMin, xMax, new ConstantDoubleVertex(c));
    }

    public TriangularVertex(int[] tensorShape, DoubleVertex xMin, double xMax, DoubleVertex c) {
        this(tensorShape, xMin, new ConstantDoubleVertex(xMax), c);
    }

    public TriangularVertex(int[] tensorShape, DoubleVertex xMin, double xMax, double c) {
        this(tensorShape, xMin, new ConstantDoubleVertex(xMax), new ConstantDoubleVertex(c));
    }

    public TriangularVertex(int[] tensorShape, double xMin, DoubleVertex xMax, DoubleVertex c) {
        this(tensorShape, new ConstantDoubleVertex(xMin), xMax, c);
    }

    public TriangularVertex(int[] tensorShape, double xMin, double xMax, DoubleVertex c) {
        this(tensorShape, new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), c);
    }

    public TriangularVertex(int[] tensorShape, double xMin, double xMax, double c) {
        this(tensorShape, new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), new ConstantDoubleVertex(c));
    }

    /**
     * One to one constructor for mapping some shape of xMin, xMax and c to a matching shaped triangular.
     *
     * @param xMin the xMin of the Triangular with either the same shape as specified for this vertex or a scalar
     * @param xMax the xMax of the Triangular with either the same shape as specified for this vertex or a scalar
     * @param c    the c of the Triangular with either the same shape as specified for this vertex or a scalar
     */
    public TriangularVertex(DoubleVertex xMin, DoubleVertex xMax, DoubleVertex c) {
        this(checkHasSingleNonScalarShapeOrAllScalar(xMin.getShape(), xMax.getShape(), c.getShape()), xMin, xMax, c);
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

    @Override
    public double logPdf(DoubleTensor value) {
        DoubleTensor xMinValues = xMin.getValue();
        DoubleTensor xMaxValues = xMax.getValue();
        DoubleTensor cValues = c.getValue();

        DoubleTensor logPdfs = Triangular.logPdf(xMinValues, xMaxValues, cValues, value);
        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return Triangular.sample(getShape(), xMin.getValue(), xMax.getValue(), c.getValue(), random);
    }
}
