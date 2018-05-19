package io.improbable.keanu.vertices.dbltensor;

import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicReference;

public class KeanuRandom {

    private static final AtomicReference<KeanuRandom> DEFAULT_RANDOM = new AtomicReference<>();

    static {
        System.setProperty("dtype", "double");

        String randomSeed = System.getProperty("io.improbable.keanu.defaultRandom.seed");

        if (randomSeed != null) {
            final long seed = Long.parseLong(randomSeed);
            DEFAULT_RANDOM.set(new KeanuRandom(seed));
        } else {
            DEFAULT_RANDOM.set(new KeanuRandom());
        }
    }

    public static KeanuRandom getDefaultRandom() {
        return DEFAULT_RANDOM.get();
    }

    public static void setDefaultRandomSeed(long seed) {
        DEFAULT_RANDOM.set(new KeanuRandom(seed));
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

    public boolean nextBoolean() {
        return nd4jRandom.nextBoolean();
    }

    public int nextInt(int maxExclusive) {
        return nd4jRandom.nextInt(maxExclusive);
    }

}
