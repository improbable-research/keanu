package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorUniform;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantDoubleTensorVertex;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;
import static java.util.Collections.singletonMap;

public class TensorUniformVertex extends TensorProbabilisticDouble {

    private final DoubleTensorVertex xMin;
    private final DoubleTensorVertex xMax;

    /**
     * @param shape desired tensor shape
     * @param xMin  inclusive
     * @param xMax  exclusive
     */
    public TensorUniformVertex(int[] shape, DoubleTensorVertex xMin, DoubleTensorVertex xMax) {

        checkTensorsMatchNonScalarShapeOrAreScalar(shape, xMin.getShape(), xMax.getShape());

        this.xMin = xMin;
        this.xMax = xMax;
        setParents(xMin, xMax);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public TensorUniformVertex(DoubleTensorVertex xMin, DoubleTensorVertex xMax) {
        this(checkHasSingleNonScalarShapeOrAllScalar(xMin.getShape(), xMax.getShape()), xMin, xMax);
    }

    public TensorUniformVertex(DoubleTensorVertex xMin, double xMax) {
        this(xMin, new ConstantDoubleTensorVertex(xMax));
    }

    public TensorUniformVertex(double xMin, DoubleTensorVertex xMax) {
        this(new ConstantDoubleTensorVertex(xMin), xMax);
    }

    public TensorUniformVertex(double xMin, double xMax) {
        this(new ConstantDoubleTensorVertex(xMin), new ConstantDoubleTensorVertex(xMax));
    }

    public TensorUniformVertex(int[] shape, DoubleTensorVertex xMin, double xMax) {
        this(shape, xMin, new ConstantDoubleTensorVertex(xMax));
    }

    public TensorUniformVertex(int[] shape, double xMin, DoubleTensorVertex xMax) {
        this(shape, new ConstantDoubleTensorVertex(xMin), xMax);
    }

    public TensorUniformVertex(int[] shape, double xMin, double xMax) {
        this(shape, new ConstantDoubleTensorVertex(xMin), new ConstantDoubleTensorVertex(xMax));
    }

    public DoubleTensorVertex getXMin() {
        return xMin;
    }

    public DoubleTensorVertex getXMax() {
        return xMax;
    }

    @Override
    public double logPdf(DoubleTensor value) {
        return TensorUniform.logPdf(xMin.getValue(), xMax.getValue(), value).sum();
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
        return TensorUniform.sample(getShape(), xMin.getValue(), xMax.getValue(), random);
    }

}
