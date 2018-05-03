package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.NDUniform;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantVertex;

import java.util.Map;

import static java.util.Collections.singletonMap;

public class NDUniformVertex extends ProbabilisticDoubleTensor {

    private final DoubleTensorVertex xMin;
    private final DoubleTensorVertex xMax;
    private final KeanuRandom random;

    public NDUniformVertex(DoubleTensorVertex xMin, DoubleTensorVertex xMax, KeanuRandom random) {
        this.xMin = xMin;
        this.xMax = xMax;
        this.random = random;
        setParents(xMin, xMax);
    }

    public NDUniformVertex(DoubleTensorVertex xMin, double xMax, KeanuRandom random) {
        this(xMin, new ConstantVertex(xMax), random);
    }

    public NDUniformVertex(double xMin, DoubleTensorVertex xMax, KeanuRandom random) {
        this(new ConstantVertex(xMin), xMax, random);
    }

    public NDUniformVertex(double xMin, double xMax, KeanuRandom random) {
        this(new ConstantVertex(xMin), new ConstantVertex(xMax), random);
    }

    public NDUniformVertex(DoubleTensorVertex xMin, DoubleTensorVertex xMax) {
        this(xMin, xMax, new KeanuRandom());
    }

    public NDUniformVertex(DoubleTensorVertex xMin, double xMax) {
        this(xMin, xMax, new KeanuRandom());
    }

    public NDUniformVertex(double xMin, DoubleTensorVertex xMax) {
        this(new ConstantVertex(xMin), xMax, new KeanuRandom());
    }

    public NDUniformVertex(double xMin, double xMax) {
        this(new ConstantVertex(xMin), new ConstantVertex(xMax), new KeanuRandom());
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
