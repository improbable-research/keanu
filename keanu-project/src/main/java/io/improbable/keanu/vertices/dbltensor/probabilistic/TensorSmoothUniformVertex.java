package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorSmoothUniform;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantTensorVertex;

import java.util.Map;

import static io.improbable.keanu.vertices.dbltensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.vertices.dbltensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;
import static java.util.Collections.singletonMap;

public class TensorSmoothUniformVertex extends ProbabilisticDoubleTensor {

    private static final double DEFAULT_EDGE_SHARPNESS = 0.01;

    private final DoubleTensorVertex xMin;
    private final DoubleTensorVertex xMax;
    private final double edgeSharpness;

    public TensorSmoothUniformVertex(int[] shape, DoubleTensorVertex xMin, DoubleTensorVertex xMax, double edgeSharpness) {

        checkTensorsMatchNonScalarShapeOrAreScalar(shape, xMin.getShape(), xMax.getShape());

        this.xMin = xMin;
        this.xMax = xMax;
        this.edgeSharpness = edgeSharpness;
        setParents(xMin, xMax);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public TensorSmoothUniformVertex(DoubleTensorVertex xMin, DoubleTensorVertex xMax, double edgeSharpness) {
        this(checkHasSingleNonScalarShapeOrAllScalar(xMin.getShape(), xMax.getShape()), xMin, xMax, edgeSharpness);
    }


    public TensorSmoothUniformVertex(DoubleTensorVertex xMin, double xMax, double edgeSharpness) {
        this(xMin, new ConstantTensorVertex(xMax), edgeSharpness);
    }

    public TensorSmoothUniformVertex(double xMin, DoubleTensorVertex xMax, double edgeSharpness) {
        this(new ConstantTensorVertex(xMin), xMax, edgeSharpness);
    }

    public TensorSmoothUniformVertex(double xMin, double xMax, double edgeSharpness) {
        this(new ConstantTensorVertex(xMin), new ConstantTensorVertex(xMax), edgeSharpness);
    }

    public TensorSmoothUniformVertex(DoubleTensorVertex xMin, DoubleTensorVertex xMax) {
        this(xMin, xMax, DEFAULT_EDGE_SHARPNESS);
    }

    public TensorSmoothUniformVertex(DoubleTensorVertex xMin, double xMax) {
        this(xMin, new ConstantTensorVertex(xMax), DEFAULT_EDGE_SHARPNESS);
    }

    public TensorSmoothUniformVertex(double xMin, DoubleTensorVertex xMax) {
        this(new ConstantTensorVertex(xMin), xMax, DEFAULT_EDGE_SHARPNESS);
    }

    public TensorSmoothUniformVertex(double xMin, double xMax) {
        this(new ConstantTensorVertex(xMin), new ConstantTensorVertex(xMax), DEFAULT_EDGE_SHARPNESS);
    }

    public TensorSmoothUniformVertex(int[] shape, DoubleTensorVertex xMin, double xMax, double edgeSharpness) {
        this(shape, xMin, new ConstantTensorVertex(xMax), edgeSharpness);
    }

    public TensorSmoothUniformVertex(int[] shape, double xMin, DoubleTensorVertex xMax, double edgeSharpness) {
        this(shape, new ConstantTensorVertex(xMin), xMax, edgeSharpness);
    }

    public TensorSmoothUniformVertex(int[] shape, double xMin, double xMax, double edgeSharpness) {
        this(shape, new ConstantTensorVertex(xMin), new ConstantTensorVertex(xMax), edgeSharpness);
    }

    public TensorSmoothUniformVertex(int[] shape, DoubleTensorVertex xMin, DoubleTensorVertex xMax) {
        this(shape, xMin, xMax, DEFAULT_EDGE_SHARPNESS);
    }

    public TensorSmoothUniformVertex(int[] shape, DoubleTensorVertex xMin, double xMax) {
        this(shape, xMin, new ConstantTensorVertex(xMax), DEFAULT_EDGE_SHARPNESS);
    }

    public TensorSmoothUniformVertex(int[] shape, double xMin, DoubleTensorVertex xMax) {
        this(shape, new ConstantTensorVertex(xMin), xMax, DEFAULT_EDGE_SHARPNESS);
    }

    public TensorSmoothUniformVertex(int[] shape, double xMin, double xMax) {
        this(shape, new ConstantTensorVertex(xMin), new ConstantTensorVertex(xMax), DEFAULT_EDGE_SHARPNESS);
    }

    @Override
    public double logPdf(DoubleTensor value) {
        final DoubleTensor min = xMin.getValue();
        final DoubleTensor max = xMax.getValue();
        final DoubleTensor shoulderWidth = (max.minus(min)).times(this.edgeSharpness);
        final DoubleTensor density = TensorSmoothUniform.pdf(min, max, shoulderWidth, value);
        return density.log().sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        final DoubleTensor min = xMin.getValue();
        final DoubleTensor max = xMax.getValue();
        final DoubleTensor shoulderWidth = (max.minus(min)).times(this.edgeSharpness);
        final DoubleTensor dPdfdx = TensorSmoothUniform.dlnPdf(min, max, shoulderWidth, value);
        final DoubleTensor density = TensorSmoothUniform.pdf(min, max, shoulderWidth, value);
        final DoubleTensor dlogPdfdx = dPdfdx.div(density);

        return singletonMap(getId(), dlogPdfdx);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return TensorSmoothUniform.sample(getShape(), xMin.getValue(), xMax.getValue(), this.edgeSharpness, random);
    }
}
