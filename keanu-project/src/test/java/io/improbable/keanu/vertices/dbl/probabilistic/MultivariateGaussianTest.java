package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethodMultiVariate;
import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.sampleUnivariateMethodMatchesLogProbMethod;
import static org.junit.Assert.assertEquals;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.continuous.MultivariateGaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.junit.Before;
import org.junit.Test;

public class MultivariateGaussianTest {

  KeanuRandom random;

  @Before
  public void setup() {
    random = new KeanuRandom(1);
  }

  @Test
  public void samplingFromUnivariateGaussianMatchesLogDensity() {
    MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(0, 1);

    double from = -2.;
    double to = 2.;
    double bucketSize = 0.05;

    sampleUnivariateMethodMatchesLogProbMethod(mvg, from, to, bucketSize, 1e-2, random, 1000000);
  }

  @Test
  public void univariateGaussianMatchesLogDensityOfScalar() {
    MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(5, 1);

    double expectedDensity = new NormalDistribution(5.0, 1).logDensity(0.5);
    double density = mvg.logPdf(Nd4jDoubleTensor.scalar(0.5));

    assertEquals(expectedDensity, density, 1e-2);
  }

  @Test
  public void bivariateGaussianMatchesLogDensityOfVector() {
    DoubleVertex mu =
        ConstantVertex.of(new Nd4jDoubleTensor(new double[] {2, 3}, new int[] {2, 1}));

    MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, 1);

    double expectedDensity1 = new NormalDistribution(2, 1).logDensity(8);
    double expectedDensity2 = new NormalDistribution(3, 1).logDensity(10);
    double expectedDensity = expectedDensity1 + expectedDensity2;

    double density = mvg.logPdf(new Nd4jDoubleTensor(new double[] {8, 10}, new int[] {2, 1}));

    assertEquals(expectedDensity, density, 0.0001);
  }

  @Test
  public void bivariateGaussianMatchesLogDensityOfScipy() {
    DoubleVertex mu =
        ConstantVertex.of(new Nd4jDoubleTensor(new double[] {1, 2}, new int[] {2, 1}));
    DoubleVertex covarianceMatrix =
        ConstantVertex.of(new Nd4jDoubleTensor(new double[] {1, 0.3, 0.3, 0.6}, new int[] {2, 2}));

    MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, covarianceMatrix);
    double density = mvg.logPdf(new Nd4jDoubleTensor(new double[] {0.5, 0.4}, new int[] {2, 1}));
    double expected = -3.6874792995813834;

    assertEquals(expected, density, 0.001);
  }

  @Test
  public void multivariateGaussianMatchesLogDensityOfScipy() {
    DoubleVertex mu =
        ConstantVertex.of(new Nd4jDoubleTensor(new double[] {1, 2, 3}, new int[] {3, 1}));

    DoubleVertex covarianceMatrix =
        ConstantVertex.of(
            new Nd4jDoubleTensor(
                new double[] {
                  1.0, 0.3, 0.3,
                  0.3, 0.8, 0.3,
                  0.3, 0.3, 0.6
                },
                new int[] {3, 3}));

    MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, covarianceMatrix);
    double density =
        mvg.logPdf(new Nd4jDoubleTensor(new double[] {0.2, 0.3, 0.4}, new int[] {3, 1}));
    double expected = -8.155504532016181;

    assertEquals(expected, density, 0.001);
  }

  @Test
  public void gaussianSampleMethodMatchesLogProbMethod() {
    DoubleVertex mu =
        ConstantVertex.of(new Nd4jDoubleTensor(new double[] {0, 0}, new int[] {2, 1}));

    MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, 1);

    double from = -1.;
    double to = 1.;
    double bucketSize = 0.05;

    sampleMethodMatchesLogProbMethodMultiVariate(
        mvg, from, to, bucketSize, 0.01, 100000, random, bucketSize * bucketSize, false);
  }

  @Test(expected = IllegalArgumentException.class)
  public void whenYouSampleYouMustMatchMusShape() {
    DoubleTensor mu = new Nd4jDoubleTensor(new double[] {0, 0}, new int[] {2, 1});
    DoubleTensor sigma = new Nd4jDoubleTensor(new double[] {1}, new int[] {1});

    ContinuousDistribution mvg = MultivariateGaussian.withParameters(mu, sigma);
    mvg.sample(new int[] {2, 2}, KeanuRandom.getDefaultRandom());
  }
}
