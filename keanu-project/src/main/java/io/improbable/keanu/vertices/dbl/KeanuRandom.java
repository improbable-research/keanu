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
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicReference;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

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

  public DoubleTensor nextDouble(int[] shape) {
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

    while ((randomValue = nextDouble()) == 0.0) ;

    return randomValue;
  }

  public DoubleTensor nextGaussian(int[] shape) {
    if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
      return new ScalarDoubleTensor(nextGaussian());
    } else {
      return new Nd4jDoubleTensor(doubleNextGaussian(shape));
    }
  }

  public DoubleTensor nextGamma(int[] shape, DoubleTensor theta, DoubleTensor k) {
    return Gamma.withParameters(theta, k).sample(shape, this);
  }

  public DoubleTensor nextLaplace(int[] shape, DoubleTensor mu, DoubleTensor beta) {
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

  public IntegerTensor nextInt(int[] shape) {
    return new Nd4jIntegerTensor(doubleNextInt(shape));
  }

  public IntegerTensor nextPoisson(int[] shape, DoubleTensor mu) {
    return Poisson.withParameters(mu).sample(shape, this);
  }

  public int nextInt(int maxExclusive) {
    return nd4jRandom.nextInt(maxExclusive);
  }

  private INDArray doubleNextInt(int[] shape) {
    Nd4j.setDataType(bufferType);
    return nd4jRandom.nextInt(shape);
  }

  private INDArray doubleNextDouble(int[] shape) {
    Nd4j.setDataType(bufferType);
    return nd4jRandom.nextDouble(shape);
  }

  private INDArray doubleNextGaussian(int[] shape) {
    Nd4j.setDataType(bufferType);
    return nd4jRandom.nextGaussian(shape);
  }
}
