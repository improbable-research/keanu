package io.improbable.keanu.vertices.dbl;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicReference;

import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;

import io.improbable.keanu.distributions.continuous.DistributionOfType;
import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.tensor.dbl.ScalarDoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.intgr.Nd4jIntegerTensor;
import io.improbable.keanu.vertices.dbl.probabilistic.DistributionVertexBuilder;

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
            return new ScalarDoubleTensor(nextDouble());
        } else {
            return new Nd4jDoubleTensor(nd4jRandom.nextDouble(shape));
        }
    }

    public double nextDouble() {
        return nd4jRandom.nextDouble();
    }

    public double nextDouble(double min, double max) {
        return nd4jRandom.nextDouble() * (max - min) + min;
    }

    public DoubleTensor nextGaussian(int[] shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new ScalarDoubleTensor(nextGaussian());
        } else {
            return new Nd4jDoubleTensor(nd4jRandom.nextGaussian(shape));
        }
    }

    public DoubleTensor nextGamma(int[] shape, DoubleTensor a, DoubleTensor theta, DoubleTensor k) {
        return DistributionOfType.gamma(a, theta, k).sample(shape, this);
    }

    public DoubleTensor nextLaplace(int[] shape, DoubleTensor mu, DoubleTensor beta) {
        return DistributionOfType.laplace(mu, beta).sample(shape, this);
    }

    public double nextGaussian() {
        return nd4jRandom.nextGaussian();
    }

    public double nextGaussian(double mu, double sigma) {
        return nd4jRandom.nextGaussian() * sigma + mu;
    }

    public boolean nextBoolean() {
        return nd4jRandom.nextBoolean();
    }

    public IntegerTensor nextInt(int[] shape) {
        return new Nd4jIntegerTensor(nd4jRandom.nextInt(shape));
    }

    public IntegerTensor nextPoisson(int[] shape, DoubleTensor mu) {
        return new DistributionVertexBuilder()
            .shaped(shape)
            .withInput(ParameterName.MU, mu)
            .poisson().sample(this);

    }

    public int nextInt(int maxExclusive) {
        return nd4jRandom.nextInt(maxExclusive);
    }

}
