package io.improbable.keanu.distributions.continuous;

import static io.improbable.keanu.distributions.dual.Diffs.BETA;
import static io.improbable.keanu.distributions.dual.Diffs.MU;
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
 * 5.1.12 page 25"
 */
public class Laplace implements ContinuousDistribution {

    private final DoubleTensor location;
    private final DoubleTensor scale;

    /**
     * @param location shifts the distribution
     * @param scale    stretches/shrinks the distribution, must be greater than 0
     * @return an instance of {@link ContinuousDistribution}
     */
    public static ContinuousDistribution withParameters(DoubleTensor location, DoubleTensor scale) {
        return new Laplace(location, scale);
    }

    private Laplace(DoubleTensor location, DoubleTensor scale) {
        this.location = location;
        this.scale = scale;
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
        Tensor.FlattenedView<Double> locationWrapped = location.getFlattenedView();
        Tensor.FlattenedView<Double> scaleWrapped = scale.getFlattenedView();

        int length = ArrayUtil.prod(shape);
        double[] samples = new double[length];
        for (int i = 0; i < length; i++) {
            samples[i] = sample(locationWrapped.getOrScalar(i), scaleWrapped.getOrScalar(i), random);
        }

        return DoubleTensor.create(samples, shape);
    }

    private static double sample(double location, double scale, KeanuRandom random) {
        if (scale <= 0.0) {
            throw new IllegalArgumentException("Invalid value for scale: " + scale);
        }
        if (random.nextDouble() > 0.5) {
            return location + scale * Math.log(random.nextDouble());
        } else {
            return location - scale * Math.log(random.nextDouble());
        }
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor locationMinusXAbsNegDivScale = location.minus(x).abs().divInPlace(scale);
        final DoubleTensor logTwoScale = scale.times(2).logInPlace();
        return  locationMinusXAbsNegDivScale.plusInPlace(logTwoScale).unaryMinus();
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor locationMinusX = location.minus(x);
        final DoubleTensor locationMinusXAbs = locationMinusX.abs();

        final DoubleTensor denominator =  locationMinusXAbs.times(scale);

        final DoubleTensor dLogPdx = locationMinusX.divInPlace(denominator);
        final DoubleTensor dLogPdlocation = x.minus(location).divInPlace(denominator);
        final DoubleTensor dLogPdscale =  locationMinusXAbs.minusInPlace(scale).divInPlace(scale.pow(2));

        return new Diffs()
            .put(MU, dLogPdlocation)
            .put(BETA, dLogPdscale)
            .put(X, dLogPdx);
    }

}