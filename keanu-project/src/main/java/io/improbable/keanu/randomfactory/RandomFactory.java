package io.improbable.keanu.randomfactory;

import io.improbable.keanu.vertices.dbl.KeanuRandom;

public interface RandomFactory<T> {

    void setRandom(KeanuRandom random);

    T nextDouble(double min, double max);

    T nextConstant(double value);

    T nextGaussian(T mu, T sigma);

    T nextGaussian(double mu, T sigma);

    T nextGaussian(T mu, double sigma);

    T nextGaussian(double mu, double sigma);

    default T nextGaussian() {
        return nextGaussian(0.0, 1.0);
    }

}
