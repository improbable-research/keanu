package io.improbable.keanu.randomFactory;

import java.util.Random;

public interface RandomFactory<T> {

    void setRandom(Random random);

    T nextDouble(double min, double max);

    T nextGaussian(double mu, double sigma);

    default T nextGaussian() {
        return nextGaussian(0.0, 1.0);
    }

}
