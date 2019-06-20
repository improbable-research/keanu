package io.improbable.keanu;

import io.improbable.keanu.distributions.continuous.Gamma;
import io.improbable.keanu.distributions.continuous.Laplace;
import io.improbable.keanu.distributions.discrete.Poisson;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;

import java.util.concurrent.atomic.AtomicReference;

import static io.improbable.keanu.tensor.TensorShape.getLengthAsInt;

public class KeanuRandom {

    private static final AtomicReference<KeanuRandom> DEFAULT_RANDOM = new AtomicReference<>();

    static {

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

    private final RandomGenerator random;

    public KeanuRandom() {
        this(System.currentTimeMillis());
    }

    public KeanuRandom(long seed) {
        random = new SynchronizedRandomGenerator(new MersenneTwister(seed));
    }

    public DoubleTensor nextDouble(long[] shape) {
        return DoubleTensor.create(nextDoubleBuffer(getLengthAsInt(shape)), shape);
    }

    public double nextDouble() {
        return random.nextDouble();
    }

    public double nextDouble(double min, double max) {
        return random.nextDouble() * (max - min) + min;
    }

    public double nextDoubleNonZero() {
        double randomValue;

        while ((randomValue = nextDouble()) == 0.0) ;

        return randomValue;
    }

    public DoubleTensor nextGaussian(long[] shape) {
        return DoubleTensor.create(nextGaussianBuffer(getLengthAsInt(shape)), shape);
    }

    public DoubleTensor nextGaussian(long[] shape, DoubleTensor mu, DoubleTensor sigma) {
        return nextGaussian(shape).timesInPlace(sigma).plusInPlace(mu);
    }

    public DoubleTensor nextGamma(long[] shape, DoubleTensor theta, DoubleTensor k) {
        return Gamma.withParameters(theta, k).sample(shape, this);
    }

    public DoubleTensor nextLaplace(long[] shape, DoubleTensor mu, DoubleTensor beta) {
        return Laplace.withParameters(mu, beta).sample(shape, this);
    }

    public double nextGaussian() {
        return random.nextGaussian();
    }

    public double nextGaussian(double mu, double sigma) {
        return random.nextGaussian() * sigma + mu;
    }

    public boolean nextBoolean() {
        return random.nextBoolean();
    }

    public IntegerTensor nextInt(long[] shape) {
        return IntegerTensor.create(nextIntBuffer(getLengthAsInt(shape)), shape);
    }

    public IntegerTensor nextPoisson(long[] shape, DoubleTensor mu) {
        return Poisson.withParameters(mu).sample(shape, this);
    }

    public int nextInt(int maxExclusive) {
        return random.nextInt(maxExclusive);
    }

    public int nextInt() {
        return random.nextInt();
    }

    private int[] nextIntBuffer(int length) {
        int[] buffer = new int[length];
        for (int i = 0; i < length; i++) {
            buffer[i] = nextInt();
        }

        return buffer;
    }

    private double[] nextDoubleBuffer(int length) {
        double[] buffer = new double[length];
        for (int i = 0; i < length; i++) {
            buffer[i] = nextDouble();
        }

        return buffer;
    }

    private double[] nextGaussianBuffer(int length) {
        double[] buffer = new double[length];
        for (int i = 0; i < length; i++) {
            buffer[i] = nextGaussian();
        }

        return buffer;
    }

}
