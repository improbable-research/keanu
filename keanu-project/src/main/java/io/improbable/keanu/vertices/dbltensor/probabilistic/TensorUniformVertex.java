package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.NDUniform;
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
    private final KeanuRandom random;

    public TensorUniformVertex(int[] shape, DoubleTensorVertex xMin, DoubleTensorVertex xMax, KeanuRandom random) {

        checkParentShapes(shape, xMin.getValue(), xMax.getValue());

        this.xMin = xMin;
        this.xMax = xMax;
        this.random = random;
        setParents(xMin, xMax);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public TensorUniformVertex(DoubleTensorVertex xMin, DoubleTensorVertex xMax, KeanuRandom random) {
        this(xMin.getValue().getShape(), xMin, xMax, random);
    }

    public TensorUniformVertex(DoubleTensorVertex xMin, double xMax, KeanuRandom random) {
        this(xMin.getValue().getShape(), xMin, new ConstantTensorVertex(xMax), random);
    }

    public TensorUniformVertex(double xMin, DoubleTensorVertex xMax, KeanuRandom random) {
        this(xMax.getValue().getShape(), new ConstantTensorVertex(xMin), xMax, random);
    }

    public TensorUniformVertex(double xMin, double xMax, KeanuRandom random) {
        this(new int[]{1, 1}, new ConstantTensorVertex(xMin), new ConstantTensorVertex(xMax), random);
    }

    public TensorUniformVertex(DoubleTensorVertex xMin, DoubleTensorVertex xMax) {
        this(getShapeProposal(xMin.getValue(), xMax.getValue()), xMin, xMax, new KeanuRandom());
    }

    public TensorUniformVertex(DoubleTensorVertex xMin, double xMax) {
        this(xMin, xMax, new KeanuRandom());
    }

    public TensorUniformVertex(double xMin, DoubleTensorVertex xMax) {
        this(xMax.getValue().getShape(), new ConstantTensorVertex(xMin), xMax, new KeanuRandom());
    }

    public TensorUniformVertex(double xMin, double xMax) {
        this(new int[]{1, 1}, new ConstantTensorVertex(xMin), new ConstantTensorVertex(xMax), new KeanuRandom());
    }

    public DoubleTensorVertex getXMin() {
        return xMin;
    }

    public DoubleTensorVertex getXMax() {
        return xMax;
    }

    @Override
    public double logPdf(DoubleTensor value) {
        return NDUniform.pdf(xMin.getValue(), xMax.getValue(), value).log().sum();
    }

    @Override
    public Map<String, DoubleTensor> dLogPdf(DoubleTensor value) {
        //TODO: add infinite gradient where invalid
        return singletonMap(getId(), DoubleTensor.zeros(this.xMax.getValue().getShape()));
    }

    @Override
    public DoubleTensor sample() {
        return NDUniform.sample(xMin.getValue(), xMax.getValue(), random);
    }


}
