package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorTriangular;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantDoubleTensorVertex;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class TensorTriangularVertex extends TensorProbabilisticDouble {

    private final DoubleTensorVertex xMin;
    private final DoubleTensorVertex xMax;
    private final DoubleTensorVertex c;

    /**
     * One xMin, xMax, c or all three driving an arbitrarily shaped tensor of Triangular
     *
     * @param shape the desired shape of the vertex
     * @param xMin  the a of the Triangular with either the same shape as specified for this vertex or a scalar
     * @param xMax  the b of the Triangular with either the same shape as specified for this vertex or a scalar
     * @param c     the c of the Triangular with either the same shape as specified for this vertex or a scalar
     */
    public TensorTriangularVertex(int[] shape, DoubleTensorVertex xMin, DoubleTensorVertex xMax, DoubleTensorVertex c) {

        checkTensorsMatchNonScalarShapeOrAreScalar(shape, xMin.getShape(), xMax.getShape(), c.getShape());

        this.xMin = xMin;
        this.xMax = xMax;
        this.c = c;
        setParents(xMin, xMax, c);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public TensorTriangularVertex(int[] shape, DoubleTensorVertex xMin, DoubleTensorVertex xMax, double c) {
        this(shape, xMin, xMax, new ConstantDoubleTensorVertex(c));
    }

    public TensorTriangularVertex(int[] shape, DoubleTensorVertex xMin, double xMax, DoubleTensorVertex c) {
        this(shape, xMin, new ConstantDoubleTensorVertex(xMax), c);
    }

    public TensorTriangularVertex(int[] shape, DoubleTensorVertex xMin, double xMax, double c) {
        this(shape, xMin, new ConstantDoubleTensorVertex(xMax), new ConstantDoubleTensorVertex(c));
    }

    public TensorTriangularVertex(int[] shape, double xMin, DoubleTensorVertex xMax, DoubleTensorVertex c) {
        this(shape, new ConstantDoubleTensorVertex(xMin), xMax, c);
    }

    public TensorTriangularVertex(int[] shape, double xMin, double xMax, DoubleTensorVertex c) {
        this(shape, new ConstantDoubleTensorVertex(xMin), new ConstantDoubleTensorVertex(xMax), c);
    }

    public TensorTriangularVertex(int[] shape, double xMin, double xMax, double c) {
        this(shape, new ConstantDoubleTensorVertex(xMin), new ConstantDoubleTensorVertex(xMax), new ConstantDoubleTensorVertex(c));
    }

    /**
     * One to one constructor for mapping some shape of xMin, xMax and c to a matching shaped triangular.
     *
     * @param xMin the xMin of the Triangular with either the same shape as specified for this vertex or a scalar
     * @param xMax the xMax of the Triangular with either the same shape as specified for this vertex or a scalar
     * @param c    the c of the Triangular with either the same shape as specified for this vertex or a scalar
     */
    public TensorTriangularVertex(DoubleTensorVertex xMin, DoubleTensorVertex xMax, DoubleTensorVertex c) {
        this(checkHasSingleNonScalarShapeOrAllScalar(xMin.getShape(), xMax.getShape(), c.getShape()), xMin, xMax, c);
    }

    public TensorTriangularVertex(DoubleTensorVertex xMin, DoubleTensorVertex xMax, double c) {
        this(xMin, xMax, new ConstantDoubleTensorVertex(c));
    }

    public TensorTriangularVertex(DoubleTensorVertex xMin, double xMax, DoubleTensorVertex c) {
        this(xMin, new ConstantDoubleTensorVertex(xMax), c);
    }

    public TensorTriangularVertex(DoubleTensorVertex xMin, double xMax, double c) {
        this(xMin, new ConstantDoubleTensorVertex(xMax), new ConstantDoubleTensorVertex(c));
    }

    public TensorTriangularVertex(double xMin, DoubleTensorVertex xMax, DoubleTensorVertex c) {
        this(new ConstantDoubleTensorVertex(xMin), xMax, c);
    }

    public TensorTriangularVertex(double xMin, double xMax, DoubleTensorVertex c) {
        this(new ConstantDoubleTensorVertex(xMin), new ConstantDoubleTensorVertex(xMax), c);
    }

    public TensorTriangularVertex(double xMin, double xMax, double c) {
        this(new ConstantDoubleTensorVertex(xMin), new ConstantDoubleTensorVertex(xMax), new ConstantDoubleTensorVertex(c));
    }


    @Override
    public double logPdf(DoubleTensor value) {
        DoubleTensor xMinValues = xMin.getValue();
        DoubleTensor xMaxValues = xMax.getValue();
        DoubleTensor cValues = c.getValue();

        DoubleTensor logPdfs = TensorTriangular.logPdf(xMinValues, xMaxValues, cValues, value);
        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return TensorTriangular.sample(getShape(), xMin.getValue(), xMax.getValue(), c.getValue(), random);
    }
}
