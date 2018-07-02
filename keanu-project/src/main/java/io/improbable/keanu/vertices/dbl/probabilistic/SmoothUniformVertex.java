package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.SmoothUniform;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;
import static java.util.Collections.singletonMap;

public class SmoothUniformVertex extends ProbabilisticDouble {

    private static final double DEFAULT_EDGE_SHARPNESS = 0.01;

    private final DoubleVertex xMin;
    private final DoubleVertex xMax;
    private final double edgeSharpness;

    /**
     * One xMin or Xmax or both driving an arbitrarily shaped tensor of Smooth Uniform
     *
     * @param tensorShape   the desired shape of the vertex
     * @param xMin          the xMin of the Smooth Uniform with either the same shape as specified for this vertex or a scalar
     * @param xMax          the xMax of the Smooth Uniform with either the same shape as specified for this vertex or a scalar
     * @param edgeSharpness the edge sharpness of the Smooth Uniform
     */
    public SmoothUniformVertex(int[] tensorShape, DoubleVertex xMin, DoubleVertex xMax, double edgeSharpness) {

        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, xMin.getShape(), xMax.getShape());

        this.xMin = xMin;
        this.xMax = xMax;
        this.edgeSharpness = edgeSharpness;
        setParents(xMin, xMax);
        setValue(DoubleTensor.placeHolder(tensorShape));
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
        this(checkHasSingleNonScalarShapeOrAllScalar(xMin.getShape(), xMax.getShape()), xMin, xMax, edgeSharpness);
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

    public SmoothUniformVertex(int[] tensorShape, DoubleVertex xMin, double xMax, double edgeSharpness) {
        this(tensorShape, xMin, new ConstantDoubleVertex(xMax), edgeSharpness);
    }

    public SmoothUniformVertex(int[] tensorShape, double xMin, DoubleVertex xMax, double edgeSharpness) {
        this(tensorShape, new ConstantDoubleVertex(xMin), xMax, edgeSharpness);
    }

    public SmoothUniformVertex(int[] tensorShape, double xMin, double xMax, double edgeSharpness) {
        this(tensorShape, new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), edgeSharpness);
    }

    public SmoothUniformVertex(int[] tensorShape, DoubleVertex xMin, DoubleVertex xMax) {
        this(tensorShape, xMin, xMax, DEFAULT_EDGE_SHARPNESS);
    }

    public SmoothUniformVertex(int[] tensorShape, DoubleVertex xMin, double xMax) {
        this(tensorShape, xMin, new ConstantDoubleVertex(xMax), DEFAULT_EDGE_SHARPNESS);
    }

    public SmoothUniformVertex(int[] tensorShape, double xMin, DoubleVertex xMax) {
        this(tensorShape, new ConstantDoubleVertex(xMin), xMax, DEFAULT_EDGE_SHARPNESS);
    }

    public SmoothUniformVertex(int[] tensorShape, double xMin, double xMax) {
        this(tensorShape, new ConstantDoubleVertex(xMin), new ConstantDoubleVertex(xMax), DEFAULT_EDGE_SHARPNESS);
    }

    @Override
    public double logPdf(DoubleTensor value) {
        final DoubleTensor min = xMin.getValue();
        final DoubleTensor max = xMax.getValue();
        final DoubleTensor shoulderWidth = (max.minus(min)).timesInPlace(this.edgeSharpness);
        final DoubleTensor density = SmoothUniform.pdf(min, max, shoulderWidth, value);
        return density.logInPlace().sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        final DoubleTensor min = xMin.getValue();
        final DoubleTensor max = xMax.getValue();
        final DoubleTensor shoulderWidth = (max.minus(min)).timesInPlace(this.edgeSharpness);
        final DoubleTensor dPdfdx = SmoothUniform.dlnPdf(min, max, shoulderWidth, value);
        final DoubleTensor density = SmoothUniform.pdf(min, max, shoulderWidth, value);
        final DoubleTensor dlogPdfdx = dPdfdx.divInPlace(density);

        return singletonMap(getId(), dlogPdfdx);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return SmoothUniform.sample(getShape(), xMin.getValue(), xMax.getValue(), this.edgeSharpness, random);
    }
}
