package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;

import static io.improbable.keanu.distributions.hyperparam.Diffs.K;
import static io.improbable.keanu.distributions.hyperparam.Diffs.THETA;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

public class Gamma implements ContinuousDistribution {

    private static final double M_E = 0.577215664901532860606512090082;
    private final DoubleTensor theta;
    private final DoubleTensor k;

    /**
     * @param theta scale
     * @param k     shape
     * @return a new ContinuousDistribution object
     */
    public static Gamma withParameters(DoubleTensor theta, DoubleTensor k) {
        return new Gamma(theta, k);
    }

    private Gamma(DoubleTensor theta, DoubleTensor k) {
        this.theta = theta;
        this.k = k;
    }

    @Override
    public DoubleTensor sample(long[] shape, KeanuRandom random) {
        long[] broadcastedShape = TensorShape.getBroadcastResultShape(shape, theta.getShape(), k.getShape());
        Tensor.FlattenedView<Double> thetaWrapped = theta.broadcast(broadcastedShape).getFlattenedView();
        Tensor.FlattenedView<Double> kWrapped = k.broadcast(broadcastedShape).getFlattenedView();

        int length = TensorShape.getLengthAsInt(broadcastedShape);
        double[] samples = new double[length];
        for (int i = 0; i < length; i++) {
            samples[i] = sample(thetaWrapped.get(i), kWrapped.get(i), random);
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

    /**
     * @param lambda shape
     * @param random source of randomness
     * @return a random number from the Exponential distribution
     */
    private static double exponentialSample(double lambda, KeanuRandom random) {
        if (lambda <= 0.0) {
            throw new IllegalArgumentException("Invalid value for b");
        }
        return -lambda * Math.log(random.nextDouble());
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor xOverTheta = x.div(theta);
        final DoubleTensor kLnTheta = k.times(theta.log());
        final DoubleTensor kMinus1LogX = k.minus(1.).timesInPlace(x.log());
        final DoubleTensor lgammaK = k.logGamma();
        return kMinus1LogX.minusInPlace(lgammaK).minusInPlace(xOverTheta).minusInPlace(kLnTheta);
    }

    public static DoubleVertex logProbOutput(DoublePlaceholderVertex x, DoublePlaceholderVertex theta, DoublePlaceholderVertex k) {
        final DoubleVertex xOverTheta = x.div(theta);
        final DoubleVertex kLnTheta = k.times(theta.log());
        final DoubleVertex kMinus1LogX = k.minus(1.).times(x.log());
        final DoubleVertex lgammaK = k.logGamma();
        return kMinus1LogX.minus(lgammaK).minus(xOverTheta).minus(kLnTheta);
    }

    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor dLogPdx = k.minus(1.).divInPlace(x).minusInPlace(theta.reciprocal());
        final DoubleTensor dLogPdtheta = theta.times(k).plusInPlace(x.unaryMinus()).divInPlace(theta.pow(2.)).unaryMinusInPlace();
        final DoubleTensor dLogPdk = x.log().minusInPlace(theta.log()).minusInPlace(k.digamma());

        return new Diffs()
            .put(THETA, dLogPdtheta)
            .put(K, dLogPdk)
            .put(X, dLogPdx);
    }

}
