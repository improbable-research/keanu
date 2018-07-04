package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Uniform;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;
import static java.util.Collections.singletonMap;

public class UniformVertex extends ProbabilisticDouble {

    private final DoubleVertex xMin;
    private final DoubleVertex xMax;

    /**
     * @param tensorShape desired tensor shape
     * @param xMin        inclusive
     * @param xMax        exclusive
     */
    public UniformVertex(int[] tensorShape, DoubleVertex xMin, DoubleVertex xMax) {

        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, xMin.getShape(), xMax.getShape());

        this.xMin = xMin;
        this.xMax = xMax;
        setParents(xMin, xMax);
        setValue(DoubleTensor.placeHolder(tensorShape));
    }

    public UniformVertex(DoubleVertex xMin, DoubleVertex xMax) {
        this(checkHasSingleNonScalarShapeOrAllScalar(xMin.getShape(), xMax.getShape()), xMin, xMax);
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

    public UniformVertex(int[] tensorShape, DoubleVertex xMin, double xMax) {
        this(tensorShape, xMin, new ConstantDoubleVertex(xMax));
    }

    public UniformVertex(int[] tensorShape, double xMin, DoubleVertex xMax) {
        this(tensorShape, new ConstantDoubleVertex(xMin), xMax);
    }

    public UniformVertex(int[] tensorShape, double xMin, double xMax) {
        this(tensorShape, new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax));
    }

    public DoubleVertex getXMin() {
        return xMin;
    }

    public DoubleVertex getXMax() {
        return xMax;
    }

    @Override
    public double logPdf(DoubleTensor value) {
        return Uniform.logPdf(xMin.getValue(), xMax.getValue(), value).sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {

        DoubleTensor dlogPdf = DoubleTensor.zeros(this.xMax.getShape());
        dlogPdf = dlogPdf.setWithMaskInPlace(value.getGreaterThanMask(xMax.getValue()), Double.NEGATIVE_INFINITY);
        dlogPdf = dlogPdf.setWithMaskInPlace(value.getLessThanOrEqualToMask(xMin.getValue()), Double.POSITIVE_INFINITY);

        return singletonMap(getId(), dlogPdf);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return Uniform.sample(getShape(), xMin.getValue(), xMax.getValue(), random);
    }

}
