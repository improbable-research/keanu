package io.improbable.keanu.distributions.continuous;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

import static io.improbable.keanu.distributions.dual.Diffs.K;
import static io.improbable.keanu.distributions.dual.Diffs.THETA;
import static io.improbable.keanu.distributions.dual.Diffs.X;

import org.nd4j.linalg.util.ArrayUtil;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

/**
 * @see "Computer Generation of Statistical Distributions
 * by Richard Saucier,
 * ARL-TR-2168 March 2000,
 * 5.1.11 page 23"
 */
public class Gamma implements ContinuousDistribution {

    private static final double M_E = 0.577215664901532860606512090082;

    private final DoubleTensor scale;
    private final DoubleTensor distributionShape;

    /**
     * @param scale             stretches/shrinks the distribution, must be greater than 0
     * @param distributionShape shape parameter (not to be confused with tensor shape)
     * @return an instance of {@link ContinuousDistribution}
     */
    public static ContinuousDistribution withParameters(DoubleTensor scale, DoubleTensor distributionShape) {
        return new Gamma(scale, distributionShape);
    }

    private Gamma(DoubleTensor scale, DoubleTensor distributionShape) {
        this.scale = scale;
        this.distributionShape = distributionShape;
    }

    /**
     * @param shape  an integer array describing the shape of the tensors to be sampled
     * @param random {@link KeanuRandom}
     * @return an instance of {@link DoubleTensor}
     * @throws IllegalArgumentException if <code>scale</code> passed to {@link #withParameters(DoubleTensor location, DoubleTensor scale)}
     *                                  is less than or equal to 0
     */
    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        Tensor.FlattenedView<Double> thetaWrapped = scale.getFlattenedView();
        Tensor.FlattenedView<Double> kWrapped = distributionShape.getFlattenedView();

        int length = ArrayUtil.prod(shape);
        double[] samples = new double[length];
        for (int i = 0; i < length; i++) {
            samples[i] = sample(thetaWrapped.getOrScalar(i), kWrapped.getOrScalar(i), random);
        }

        return DoubleTensor.create(samples, shape);
    }

    private static double sample(double theta, double k, KeanuRandom random) {
        if (theta <= 0. || k <= 0.) {
            throw new IllegalArgumentException("Invalid value for theta or k. Theta: " + theta + ". k: " + k);
        }
        final double A = 1. / sqrt(2. * k - 1.);
        final double B = k - log(4.);
        final double Q = k + 1. / A;
        final double T = 4.5;
        final double D = 1. + log(T);
        final double C = 1. + k / M_E;

        if (k < 1.) {
            return sampleWhileKLessThanOne(C, k, theta, random);
        } else if (k == 1.0) return exponentialSample(theta, random);
        else {
            while (true) {
                double p1 = random.nextDouble();
                double p2 = random.nextDouble();
                double v = A * log(p1 / (1. - p1));
                double y = k * exp(v);
                double z = p1 * p1 * p2;
                double w = B + Q * v - y;
                if (w + D - T * z >= 0. || w >= log(z)) return theta * y;
            }
        }
    }

    private static double sampleWhileKLessThanOne(double c, double k, double theta, KeanuRandom random) {
        while (true) {
            double p = c * random.nextDouble();
            if (p > 1.) {
                double y = -log((c - p) / k);
                if (random.nextDouble() <= pow(y, k - 1.)) return theta * y;
            } else {
                double y = pow(p, 1. / k);
                if (random.nextDouble() <= exp(-y)) return theta * y;
            }
        }
    }

    private static double exponentialSample(double lambda, KeanuRandom random) {
        if (lambda <= 0.0) {
            throw new IllegalArgumentException("Invalid value for b");
        }
        return -lambda * Math.log(random.nextDouble());
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor minusXOverTheta = x.div(scale).unaryMinusInPlace();
        final DoubleTensor kLnTheta = distributionShape.times(scale.log());
        final DoubleTensor xPowKMinus1 = x.pow(distributionShape.minus(1));
        final DoubleTensor lnXToKMinus1 = (xPowKMinus1.divInPlace(distributionShape.apply(org.apache.commons.math3.special.Gamma::gamma))).logInPlace();
        return minusXOverTheta.minusInPlace(kLnTheta).plusInPlace(lnXToKMinus1);
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor dLogPdx = distributionShape.minus(1.).divInPlace(x).minusInPlace(scale.reciprocal());
        final DoubleTensor dLogPdtheta = scale.times(distributionShape).plusInPlace(x.unaryMinus()).divInPlace(scale.pow(2.)).unaryMinusInPlace();
        final DoubleTensor dLogPdk = x.log().minusInPlace(scale.log()).minusInPlace(distributionShape.apply(org.apache.commons.math3.special.Gamma::digamma));

        return new Diffs()
            .put(THETA, dLogPdtheta)
            .put(K, dLogPdk)
            .put(X, dLogPdx);
    }

}