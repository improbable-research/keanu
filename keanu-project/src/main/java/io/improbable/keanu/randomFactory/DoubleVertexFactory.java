package io.improbable.keanu.randomFactory;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

import java.util.Random;

public class DoubleVertexFactory implements RandomFactory<DoubleVertex> {

    private Random random = new Random();

    @Override
    public void setRandom(Random random) {
        this.random = random;
    }

    @Override
    public UniformVertex nextDouble(double min, double max) {
        return new UniformVertex(min, max, random);
    }

    @Override
    public ConstantDoubleVertex nextConstant(double value) {
        return new ConstantDoubleVertex(value);
    }

    @Override
    public GaussianVertex nextGaussian(double mu, double sigma) {
        return new GaussianVertex(mu, sigma, random);
    }

    @Override
    public GaussianVertex nextGaussian(DoubleVertex mu, double sigma) {
        return new GaussianVertex(mu, sigma, random);
    }

    @Override
    public GaussianVertex nextGaussian(DoubleVertex mu, DoubleVertex sigma) {
        return new GaussianVertex(mu, sigma, random);
    }

    @Override
    public DoubleVertex nextGaussian(double mu, DoubleVertex sigma) {
        return new GaussianVertex(mu, sigma, random);
    }

}
