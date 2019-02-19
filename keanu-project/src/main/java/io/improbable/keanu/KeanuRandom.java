package io.improbable.keanu;

import io.improbable.keanu.distributions.continuous.Gamma;
import io.improbable.keanu.distributions.continuous.Laplace;
import io.improbable.keanu.distributions.discrete.Poisson;
import io.improbable.keanu.tensor.INDArrayShim;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.ScalarDoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.intgr.Nd4jIntegerTensor;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicReference;

public class KeanuRandom {

    private static final AtomicReference<KeanuRandom> DEFAULT_RANDOM = new AtomicReference<>();

    static {
        INDArrayShim.startNewThreadForNd4j();

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
    private final DataBuffer.Type bufferType;

    public KeanuRandom() {
        nd4jRandom = new DefaultRandom();
        bufferType = DataBuffer.Type.DOUBLE;
    }

    public KeanuRandom(long seed) {
        nd4jRandom = new DefaultRandom(seed);
        bufferType = DataBuffer.Type.DOUBLE;
    }

    public DoubleTensor nextDouble(long[] shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new ScalarDoubleTensor(nextDouble());
        } else {
            return DoubleTensor.create(doubleNextDouble(shape), shape);
        }
    }

    public double nextDouble() {
        return nd4jRandom.nextDouble();
    }

    public double nextDouble(double min, double max) {
        return nd4jRandom.nextDouble() * (max - min) + min;
    }

    public double nextDoubleNonZero() {
        double randomValue;

        while ((randomValue = nextDouble()) == 0.0) ;

        return randomValue;
    }

    public DoubleTensor nextGaussian(long[] shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new ScalarDoubleTensor(nextGaussian());
        } else {
            return DoubleTensor.create(doubleNextGaussian(shape), shape);
        }
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
        return nd4jRandom.nextGaussian();
    }

    public double nextGaussian(double mu, double sigma) {
        return nd4jRandom.nextGaussian() * sigma + mu;
    }

    public boolean nextBoolean() {
        return nd4jRandom.nextBoolean();
    }

    public IntegerTensor nextInt(long[] shape) {
        return new Nd4jIntegerTensor(doubleNextInt(shape));
    }

    public IntegerTensor nextPoisson(long[] shape, DoubleTensor mu) {
        return Poisson.withParameters(mu).sample(shape, this);

    }

    public int nextInt(int maxExclusive) {
        return nd4jRandom.nextInt(maxExclusive);
    }

    private INDArray doubleNextInt(long[] shape) {
        Nd4j.setDataType(bufferType);
        return nd4jRandom.nextInt(shape);
    }

    private double[] doubleNextDouble(long[] shape) {
        int length = TensorShape.getLengthAsInt(shape);
        double[] buffer = new double[length];
        for (int i = 0; i < length; i++) {
            buffer[i] = nextDouble();
        }

        return buffer;
    }

    private double[] doubleNextGaussian(long[] shape) {

        int length = TensorShape.getLengthAsInt(shape);
        double[] buffer = new double[length];
        for (int i = 0; i < length; i++) {
            buffer[i] = nextGaussian();
        }

        return buffer;
    }

}
