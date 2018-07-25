package io.improbable.keanu.randomfactory;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.DistributionVertexBuilder;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.VertexOfType;

public class DoubleVertexFactory implements RandomFactory<DoubleVertex> {

    private KeanuRandom random = new KeanuRandom();

    @Override
    public void setRandom(KeanuRandom random) {
        this.random = random;
    }

    @Override
    public UniformVertex nextDouble(double min, double max) {
        UniformVertex uniformVertex = VertexOfType.uniform(min, max);
        uniformVertex.setValue(uniformVertex.sample(random));
        return uniformVertex;
    }

    @Override
    public ConstantDoubleVertex nextConstant(double value) {
        return new ConstantDoubleVertex(value);
    }

    @Override
    public GaussianVertex nextGaussian(double mu, double sigma) {
        GaussianVertex gaussianVertex = new DistributionVertexBuilder().withInput(ParameterName.MU, mu).withInput(ParameterName.SIGMA, sigma).gaussian();
        gaussianVertex.setValue(gaussianVertex.sample(random));
        return gaussianVertex;
    }

    @Override
    public GaussianVertex nextGaussian(DoubleVertex mu, double sigma) {
        GaussianVertex gaussianVertex = new DistributionVertexBuilder().withInput(ParameterName.MU, mu).withInput(ParameterName.SIGMA, sigma).gaussian();
        gaussianVertex.setValue(gaussianVertex.sample(random));
        return gaussianVertex;
    }

    @Override
    public GaussianVertex nextGaussian(DoubleVertex mu, DoubleVertex sigma) {
        GaussianVertex gaussianVertex = new DistributionVertexBuilder().withInput(ParameterName.MU, mu).withInput(ParameterName.SIGMA, sigma).gaussian();
        gaussianVertex.setValue(gaussianVertex.sample(random));
        return gaussianVertex;
    }

    @Override
    public DoubleVertex nextGaussian(double mu, DoubleVertex sigma) {
        GaussianVertex gaussianVertex = new DistributionVertexBuilder().withInput(ParameterName.MU, mu).withInput(ParameterName.SIGMA, sigma).gaussian();
        gaussianVertex.setValue(gaussianVertex.sample(random));
        return gaussianVertex;
    }

}
