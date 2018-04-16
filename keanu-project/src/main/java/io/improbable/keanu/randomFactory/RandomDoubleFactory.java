package io.improbable.keanu.randomFactory;

import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.distributions.continuous.Uniform;
import io.improbable.keanu.kotlin.ArithmeticDouble;

import java.util.Random;

public class RandomDoubleFactory implements RandomFactory<ArithmeticDouble> {

    private Random random = new Random();

    @Override
    public void setRandom(Random random) {
        this.random = random;
    }

    @Override
    public ArithmeticDouble nextDouble(double min, double max) {
        double randomDouble = Uniform.sample(min, max, random);
        return new ArithmeticDouble(randomDouble);
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
