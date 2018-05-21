package io.improbable.keanu.vertices.dbltensor;

import io.improbable.keanu.distributions.continuous.Gamma;
import io.improbable.keanu.distributions.continuous.Laplace;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.util.ArrayUtil;

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
        return new Nd4jDoubleTensor(nd4jRandom.nextDouble(shape));
    }

    public double nextDouble() {
        return nd4jRandom.nextDouble();
    }

    public DoubleTensor nextGaussian(int[] shape) {
        return new Nd4jDoubleTensor(nd4jRandom.nextGaussian(shape));
    }

    public DoubleTensor nextGamma(int[] shape, DoubleTensor a, DoubleTensor theta, DoubleTensor k) {

        DataBufferWrapper aWrapped = new DataBufferWrapper(a.getLinearView());
        DataBufferWrapper thetaWrapped = new DataBufferWrapper(theta.getLinearView());
        DataBufferWrapper kWrapped = new DataBufferWrapper(k.getLinearView());

        int length = ArrayUtil.prod(shape);
        double[] samples = new double[length];
        for (int i = 0; i < length; i++) {
            samples[i] = Gamma.sample(aWrapped.get(i), thetaWrapped.get(i), kWrapped.get(i), this);
        }

        return DoubleTensor.create(samples, shape);
    }

    public DoubleTensor nextLaplace(int[] shape, DoubleTensor mu, DoubleTensor beta) {

        DataBufferWrapper aWrapped = new DataBufferWrapper(mu.getLinearView());
        DataBufferWrapper betaWrapped = new DataBufferWrapper(beta.getLinearView());

        int length = ArrayUtil.prod(shape);
        double[] samples = new double[length];
        for (int i = 0; i < length; i++) {
            samples[i] = Laplace.sample(aWrapped.get(i), betaWrapped.get(i), this);
        }

        return DoubleTensor.create(samples, shape);
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

    private static class DataBufferWrapper {
        private final double[] buffer;

        DataBufferWrapper(double[] buffer) {
            this.buffer = buffer;
        }

        public double get(int i) {
            if (buffer.length == 1) {
                return buffer[0];
            } else {
                return buffer[i];
            }
        }
    }
}
