package io.improbable.keanu.randomfactory;

import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.distributions.continuous.Uniform;
import io.improbable.keanu.kotlin.ArithmeticDouble;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class RandomDoubleFactory implements RandomFactory<ArithmeticDouble> {

    private KeanuRandom random = new KeanuRandom();

    @Override
    public void setRandom(KeanuRandom random) {
        this.random = random;
    }

    @Override
    public ArithmeticDouble nextDouble(double min, double max) {
        double randomDouble = Uniform.sample(min, max, random);
        return new ArithmeticDouble(randomDouble);
    }

    @Override
    public ArithmeticDouble nextConstant(double value) {
        return new ArithmeticDouble(value);
    }

    @Override
    public ArithmeticDouble nextGaussian(ArithmeticDouble mu, ArithmeticDouble sigma) {
        double randomDouble = Gaussian.sample(mu.getValue(), sigma.getValue(), random);
        return new ArithmeticDouble(randomDouble);
    }

    @Override
    public ArithmeticDouble nextGaussian(double mu, ArithmeticDouble sigma) {
        double randomDouble = Gaussian.sample(mu, sigma.getValue(), random);
        return new ArithmeticDouble(randomDouble);
    }

    @Override
    public ArithmeticDouble nextGaussian(ArithmeticDouble mu, double sigma) {
        double randomDouble = Gaussian.sample(mu.getValue(), sigma, random);
        return new ArithmeticDouble(randomDouble);
    }

    @Override
    public ArithmeticDouble nextGaussian(double mu, double sigma) {
        double randomDouble = Gaussian.sample(mu, sigma, random);
        return new ArithmeticDouble(randomDouble);
    }
}
