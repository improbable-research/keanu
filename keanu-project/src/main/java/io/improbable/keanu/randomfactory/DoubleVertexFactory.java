package io.improbable.keanu.randomfactory;

import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantDoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorGaussianVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorUniformVertex;

public class DoubleVertexFactory implements RandomFactory<DoubleTensorVertex> {

    private KeanuRandom random = new KeanuRandom();

    @Override
    public void setRandom(KeanuRandom random) {
        this.random = random;
    }

    @Override
    public TensorUniformVertex nextDouble(double min, double max) {
        TensorUniformVertex uniformVertex = new TensorUniformVertex(min, max);
        uniformVertex.setValue(uniformVertex.sample(random));
        return uniformVertex;
    }

    @Override
    public ConstantDoubleTensorVertex nextConstant(double value) {
        return new ConstantDoubleTensorVertex(value);
    }

    @Override
    public TensorGaussianVertex nextGaussian(double mu, double sigma) {
        TensorGaussianVertex gaussianVertex = new TensorGaussianVertex(mu, sigma);
        gaussianVertex.setValue(gaussianVertex.sample(random));
        return gaussianVertex;
    }

    @Override
    public TensorGaussianVertex nextGaussian(DoubleTensorVertex mu, double sigma) {
        TensorGaussianVertex gaussianVertex = new TensorGaussianVertex(mu, sigma);
        gaussianVertex.setValue(gaussianVertex.sample(random));
        return gaussianVertex;
    }

    @Override
    public TensorGaussianVertex nextGaussian(DoubleTensorVertex mu, DoubleTensorVertex sigma) {
        TensorGaussianVertex gaussianVertex = new TensorGaussianVertex(mu, sigma);
        gaussianVertex.setValue(gaussianVertex.sample(random));
        return gaussianVertex;
    }

    @Override
    public DoubleTensorVertex nextGaussian(double mu, DoubleTensorVertex sigma) {
        TensorGaussianVertex gaussianVertex = new TensorGaussianVertex(mu, sigma);
        gaussianVertex.setValue(gaussianVertex.sample(random));
        return gaussianVertex;
    }

}
