package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.distributions.continuous.Gamma;
import io.improbable.keanu.distributions.continuous.Laplace;
import io.improbable.keanu.distributions.discrete.Poisson;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.tensor.dbl.ScalarDoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.intgr.Nd4jIntegerTensor;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.util.ArrayUtil;

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
            return new ScalarDoubleTensor(nextDouble());
        } else {
            return new Nd4jDoubleTensor(nd4jRandom.nextDouble(shape));
        }
    }

    public double nextDouble() {
        return nd4jRandom.nextDouble();
    }

    public DoubleTensor nextGaussian(int[] shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new ScalarDoubleTensor(nextGaussian());
        } else {
            return new Nd4jDoubleTensor(nd4jRandom.nextGaussian(shape));
        }
    }

    public DoubleTensor nextGamma(int[] shape, DoubleTensor a, DoubleTensor theta, DoubleTensor k) {

        Tensor.FlattenedView<Double> aWrapped = a.getFlattenedView();
        Tensor.FlattenedView<Double> thetaWrapped = theta.getFlattenedView();
        Tensor.FlattenedView<Double> kWrapped = k.getFlattenedView();

        int length = ArrayUtil.prod(shape);
        double[] samples = new double[length];
        for (int i = 0; i < length; i++) {
            samples[i] = Gamma.sample(aWrapped.getOrScalar(i), thetaWrapped.getOrScalar(i), kWrapped.getOrScalar(i), this);
        }

        return DoubleTensor.create(samples, shape);
    }

    public DoubleTensor nextLaplace(int[] shape, DoubleTensor mu, DoubleTensor beta) {

        Tensor.FlattenedView<Double> muWrapped = mu.getFlattenedView();
        Tensor.FlattenedView<Double> betaWrapped = beta.getFlattenedView();

        int length = ArrayUtil.prod(shape);
        double[] samples = new double[length];
        for (int i = 0; i < length; i++) {
            samples[i] = Laplace.sample(muWrapped.getOrScalar(i), betaWrapped.getOrScalar(i), this);
        }

        return DoubleTensor.create(samples, shape);
    }

    public double nextGaussian() {
        return nd4jRandom.nextGaussian();
    }

    public boolean nextBoolean() {
        return nd4jRandom.nextBoolean();
    }

    public IntegerTensor nextInt(int[] shape) {
        return new Nd4jIntegerTensor(nd4jRandom.nextInt(shape));
    }

    public IntegerTensor nextPoisson(int[] shape, DoubleTensor mu) {
        Tensor.FlattenedView<Double> muWrapped = mu.getFlattenedView();

        int length = ArrayUtil.prod(shape);
        int[] samples = new int[length];
        for (int i = 0; i < length; i++) {
            samples[i] = Poisson.sample(muWrapped.getOrScalar(i), this);
        }

        return IntegerTensor.create(samples, shape);

    }

    public int nextInt(int maxExclusive) {
        return nd4jRandom.nextInt(maxExclusive);
    }

}
