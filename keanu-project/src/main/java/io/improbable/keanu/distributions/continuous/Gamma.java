package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.nd4j.linalg.util.ArrayUtil;

import static io.improbable.keanu.tensor.Tensor.SCALAR_SHAPE;
import static io.improbable.keanu.tensor.TensorShape.concat;
import static java.lang.Math.*;

public class Gamma {

    private static final double M_E = 0.577215664901532860606512090082;

    private Gamma() {
    }

    /**
     * @param shape  shape of tensor returned
     * @param a      location
     * @param theta  scale
     * @param k      shape
     * @param random source of randomness
     * @return a random number from the Gamma distribution
     */
    public static DoubleTensor sample(int[] shape, DoubleTensor a, DoubleTensor theta, DoubleTensor k, KeanuRandom random) {

        Tensor.FlattenedView<Double> aWrapped = a.getFlattenedView();
        Tensor.FlattenedView<Double> thetaWrapped = theta.getFlattenedView();
        Tensor.FlattenedView<Double> kWrapped = k.getFlattenedView();

        int length = ArrayUtil.prod(shape);
        double[] samples = new double[length];
        for (int i = 0; i < length; i++) {
            samples[i] = sample(aWrapped.getOrScalar(i), thetaWrapped.getOrScalar(i), kWrapped.getOrScalar(i), random);
        }

        return DoubleTensor.create(samples, shape);
    }

    private static double sample(double a, double theta, double k, KeanuRandom random) {
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
            return sampleWhileKLessThanOne(C, k, a, theta, random);
        } else if (k == 1.0) return exponentialSample(a, theta, random);
        else {
            while (true) {
                double p1 = random.nextDouble();
                double p2 = random.nextDouble();
                double v = A * log(p1 / (1. - p1));
                double y = k * exp(v);
                double z = p1 * p1 * p2;
                double w = B + Q * v - y;
                if (w + D - T * z >= 0. || w >= log(z)) return a + theta * y;
            }
        }
    }

    private static double sampleWhileKLessThanOne(double c, double k, double a, double theta, KeanuRandom random) {
        while (true) {
            double p = c * random.nextDouble();
            if (p > 1.) {
                double y = -log((c - p) / k);
                if (random.nextDouble() <= pow(y, k - 1.)) return a + theta * y;
            } else {
                double y = pow(p, 1. / k);
                if (random.nextDouble() <= exp(-y)) return a + theta * y;
            }
        }
    }

    /**
     * @param location location
     * @param lambda   shape
     * @param random   source of randomness
     * @return a random number from the Exponential distribution
     */
    private static double exponentialSample(double location, double lambda, KeanuRandom random) {
        if (lambda <= 0.0) {
            throw new IllegalArgumentException("Invalid value for b");
        }
        return location - lambda * Math.log(random.nextDouble());
    }

    public static DoubleTensor logPdf(DoubleTensor location, DoubleTensor theta, DoubleTensor k, DoubleTensor x) {
        final DoubleTensor aMinusXOverTheta = location.minus(x).divInPlace(theta);
        final DoubleTensor kLnTheta = k.times(theta.log());
        final DoubleTensor xMinusAPowKMinus1 = x.minus(location).powInPlace(k.minus(1));
        final DoubleTensor lnXMinusAToKMinus1 = ((xMinusAPowKMinus1).divInPlace(k.apply(org.apache.commons.math3.special.Gamma::gamma))).logInPlace();
        return aMinusXOverTheta.minusInPlace(kLnTheta).plusInPlace(lnXMinusAToKMinus1);
    }

    public static DiffLogP dlnPdf(DoubleTensor location, DoubleTensor theta, DoubleTensor k, DoubleTensor x) {
        final DoubleTensor xMinusLocation = x.minus(location);
        final DoubleTensor locationMinusX = location.minus(x);
        final DoubleTensor kMinus1 = k.minus(1.);
        final DoubleTensor oneOverTheta = theta.reciprocal();

        DoubleTensor dLogPdx = kMinus1.div(xMinusLocation).minusInPlace(oneOverTheta);
        DoubleTensor dLogPdlocation = kMinus1.div(locationMinusX).plusInPlace(oneOverTheta);
        DoubleTensor dLogPdtheta = theta.times(k).plus(locationMinusX).divInPlace(theta.pow(2.)).unaryMinusInPlace();
        DoubleTensor dLogPdk = xMinusLocation.logInPlace().minusInPlace(theta.log()).minusInPlace(k.apply(org.apache.commons.math3.special.Gamma::digamma));

        return new DiffLogP(dLogPdlocation, dLogPdtheta, dLogPdk, dLogPdx);
    }

    public static class DiffLogP {
        public final DoubleTensor dLogPdlocation;
        public final DoubleTensor dLogPdtheta;
        public final DoubleTensor dLogPdk;
        public final DoubleTensor dLogPdx;

        public DiffLogP(DoubleTensor dLogPdlocation, DoubleTensor dLogPdtheta, DoubleTensor dLogPdk, DoubleTensor dLogPdx) {
            this.dLogPdlocation = dLogPdlocation;
            this.dLogPdtheta = dLogPdtheta;
            this.dLogPdk = dLogPdk;
            this.dLogPdx = dLogPdx;
        }
    }

}
