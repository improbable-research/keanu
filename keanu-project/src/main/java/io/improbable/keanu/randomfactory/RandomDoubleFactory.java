package io.improbable.keanu.randomfactory;

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
        double randomDouble = random.nextDouble(min, max);
        return new ArithmeticDouble(randomDouble);
    }

    @Override
    public ArithmeticDouble nextConstant(double value) {
        return new ArithmeticDouble(value);
    }

    @Override
    public ArithmeticDouble nextGaussian(ArithmeticDouble mu, ArithmeticDouble sigma) {
        double randomDouble = random.nextGaussian(mu.getValue(), sigma.getValue());
        return new ArithmeticDouble(randomDouble);
    }

    @Override
    public ArithmeticDouble nextGaussian(double mu, ArithmeticDouble sigma) {
        double randomDouble = random.nextGaussian(mu, sigma.getValue());
        return new ArithmeticDouble(randomDouble);
    }

    @Override
    public ArithmeticDouble nextGaussian(ArithmeticDouble mu, double sigma) {
        double randomDouble = random.nextGaussian(mu.getValue(), sigma);
        return new ArithmeticDouble(randomDouble);
    }

    @Override
    public ArithmeticDouble nextGaussian(double mu, double sigma) {
        double randomDouble = random.nextGaussian(mu, sigma);
        return new ArithmeticDouble(randomDouble);
    }
}
