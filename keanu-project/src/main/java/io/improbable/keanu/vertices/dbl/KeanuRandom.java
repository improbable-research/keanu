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
        /*
         * We need to load ND4J in a separate thread as on load it sets the FTZ and DAZ flags in the processor for the
         * thread that does the load.  This causes issues with Apache Math that makes use of Sub-normal values (in
         * particular to initialisation values for the BrentOptimizer).
         *
         * We have raised https://github.com/deeplearning4j/deeplearning4j/issues/6690 to address this
         */
        Thread nd4jInitThread = new Thread(new Runnable() {
            @Override
            public void run() {
                DoubleTensor a = DoubleTensor.create(1.0, 1.0);
            }
        });
        nd4jInitThread.start();
        try {
            nd4jInitThread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

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
            return new Nd4jDoubleTensor(doubleNextDouble(shape));
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

        while ((randomValue = nextDouble()) == 0.0);

        return randomValue;
    }

    public DoubleTensor nextGaussian(long[] shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new ScalarDoubleTensor(nextGaussian());
        } else {
            return new Nd4jDoubleTensor(doubleNextGaussian(shape));
        }
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

    private INDArray doubleNextDouble(long[] shape) {
        Nd4j.setDataType(bufferType);
        return nd4jRandom.nextDouble(shape);
    }

    private INDArray doubleNextGaussian(long[] shape) {
        Nd4j.setDataType(bufferType);
        return nd4jRandom.nextGaussian(shape);
    }

}
