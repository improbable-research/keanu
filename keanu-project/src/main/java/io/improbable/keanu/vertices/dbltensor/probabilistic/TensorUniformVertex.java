package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorUniform;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantTensorVertex;

import java.util.Map;

import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticVertexShaping.checkParentShapes;
import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticVertexShaping.getShapeProposal;
import static java.util.Collections.singletonMap;

public class TensorUniformVertex extends ProbabilisticDoubleTensor {

    private final DoubleTensorVertex xMin;
    private final DoubleTensorVertex xMax;

    /**
     * @param shape  desired tensor shape
     * @param xMin   inclusive
     * @param xMax   exclusive
     */
    public TensorUniformVertex(int[] shape, DoubleTensorVertex xMin, DoubleTensorVertex xMax) {

        checkParentShapes(shape, xMin.getValue(), xMax.getValue());

        this.xMin = xMin;
        this.xMax = xMax;
        setParents(xMin, xMax);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public TensorUniformVertex(DoubleTensorVertex xMin, DoubleTensorVertex xMax) {
        this(xMin.getValue().getShape(), xMin, xMax);
    }

    public TensorUniformVertex(DoubleTensorVertex xMin, double xMax) {
        this(xMin.getValue().getShape(), xMin, new ConstantTensorVertex(xMax));
    }

    public TensorUniformVertex(double xMin, DoubleTensorVertex xMax) {
        this(xMax.getValue().getShape(), new ConstantTensorVertex(xMin), xMax);
    }

    public TensorUniformVertex(double xMin, double xMax) {
        this(new int[]{1, 1}, new ConstantTensorVertex(xMin), new ConstantTensorVertex(xMax));
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

        DoubleTensor dlogPdf = DoubleTensor.zeros(this.xMax.getValue().getShape());
        dlogPdf.applyWhere(value.getGreaterThanMask(xMax.getValue()), Double.NEGATIVE_INFINITY);
        dlogPdf.applyWhere(value.getLessThanOrEqualToMask(xMin.getValue()), Double.POSITIVE_INFINITY);

        return singletonMap(getId(), dlogPdf);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return TensorUniform.sample(xMin.getValue(), xMax.getValue(), random);
    }


}
