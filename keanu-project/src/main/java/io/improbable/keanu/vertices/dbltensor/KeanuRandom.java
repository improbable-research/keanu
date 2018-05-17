package io.improbable.keanu.vertices.dbltensor;

import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;

import java.util.Arrays;

public class KeanuRandom extends java.util.Random {

    static {
        System.setProperty("dtype", "double");
    }

    private final Random nd4jRandom;

    public KeanuRandom() {
        nd4jRandom = new DefaultRandom();
    }

    public KeanuRandom(long seed) {
        nd4jRandom = new DefaultRandom(seed);
    }

    public DoubleTensor nextDouble(int[] shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new SimpleDoubleTensor(nextDouble());
        } else {
            return new Nd4jDoubleTensor(nd4jRandom.nextDouble(shape));
        }
    }

    public double nextDouble() {
        return nd4jRandom.nextDouble();
    }

    public DoubleTensor nextGaussian(int[] shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new SimpleDoubleTensor(nextGaussian());
        } else {
            return new Nd4jDoubleTensor(nd4jRandom.nextGaussian(shape));
        }
    }

    public double nextGaussian() {
        return nd4jRandom.nextGaussian();
    }
}
