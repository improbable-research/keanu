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
    private final DoubleTensor alpha;

    /**
     * @param scale    stretches/shrinks the distribution, must be greater than 0
     * @param alpha    shape parameter (not to be confused with tensor shape)
     * @return an instance of {@link ContinuousDistribution}
     */
    public static ContinuousDistribution withParameters(DoubleTensor scale, DoubleTensor alpha) {
        return new Gamma(scale, alpha);
    }

    private Gamma(DoubleTensor scale, DoubleTensor alpha) {
        this.scale = scale;
        this.alpha = alpha;
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
        Tensor.FlattenedView<Double> scaleWrapped = scale.getFlattenedView();
        Tensor.FlattenedView<Double> alphaWrapped = alpha.getFlattenedView();

        int length = ArrayUtil.prod(shape);
        double[] samples = new double[length];
        for (int i = 0; i < length; i++) {
            samples[i] = sample(scaleWrapped.getOrScalar(i), alphaWrapped.getOrScalar(i), random);
        }

        return DoubleTensor.create(samples, shape);
    }

    private static double sample(double scale, double alpha, KeanuRandom random) {
        if (scale <= 0. || alpha <= 0.) {
            throw new IllegalArgumentException("Invalid value for scale or alpha. Scale: " + scale + ". Alpha: " + alpha);
        }
        final double A = 1. / sqrt(2. * alpha - 1.);
        final double B = alpha - log(4.);
        final double Q = alpha + 1. / A;
        final double T = 4.5;
        final double D = 1. + log(T);
        final double C = 1. + alpha / M_E;

        if (alpha < 1.) {
            return sampleWhileKLessThanOne(C, alpha, scale, random);
        } else if (alpha == 1.0) {
            return exponentialSample(scale, random);
        } else {
            while (true) {
                double p1 = random.nextDouble();
                double p2 = random.nextDouble();
                double v = A * log(p1 / (1. - p1));
                double y = alpha * exp(v);
                double z = p1 * p1 * p2;
                double w = B + Q * v - y;
                if (w + D - T * z >= 0. || w >= log(z)) return scale * y;
            }
        }
    }

    private static double sampleWhileKLessThanOne(double c, double alpha, double scale, KeanuRandom random) {
        while (true) {
            double p = c * random.nextDouble();
            if (p > 1.) {
                double y = -log((c - p) / alpha);
                if (random.nextDouble() <= pow(y, alpha - 1.)) return scale * y;
            } else {
                double y = pow(p, 1. / alpha);
                if (random.nextDouble() <= exp(-y)) return scale * y;
            }
        }
    }

    private static double exponentialSample(double scale, KeanuRandom random) {
        return -scale * Math.log(random.nextDouble());
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor minusXOverTheta = x.div(scale).unaryMinusInPlace();
        final DoubleTensor kLnTheta = alpha.times(scale.log());
        final DoubleTensor xPowKMinus1 = x.pow(alpha.minus(1));
        final DoubleTensor lnXToKMinus1 = (xPowKMinus1.divInPlace(alpha.apply(org.apache.commons.math3.special.Gamma::gamma))).logInPlace();
        return minusXOverTheta.minusInPlace(kLnTheta).plusInPlace(lnXToKMinus1);
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor dLogPdx = alpha.minus(1.).divInPlace(x).minusInPlace(scale.reciprocal());
        final DoubleTensor dLogPdtheta = scale.times(alpha).plusInPlace(x.unaryMinus()).divInPlace(scale.pow(2.)).unaryMinusInPlace();
        final DoubleTensor dLogPdk = x.log().minusInPlace(scale.log()).minusInPlace(alpha.apply(org.apache.commons.math3.special.Gamma::digamma));

        return new Diffs()
        .put(THETA, dLogPdtheta)
        .put(K, dLogPdk)
        .put(X, dLogPdx);
    }

}