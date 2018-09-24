package io.improbable.keanu.distributions.continuous;

import static java.lang.Math.PI;
import static java.lang.Math.log;

import static io.improbable.keanu.distributions.dual.Diffs.T;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

/**
 * Student T Distribution
 * https://en.wikipedia.org/wiki/Student%27s_t-distribution#Sampling_distribution
 */
public class StudentT implements ContinuousDistribution {

    private static final double HALF_LOG_PI = log(PI) / 2;
    private final IntegerTensor v;

    /**
     * Computer Generation of Statistical Distributions
     * by Richard Saucier
     * ARL-TR-2168 March 2000
     * 5.1.23 page 36
     *
     * @param v Degrees of Freedom
     * @return a new ContinuousDistribution object
     */
    public static ContinuousDistribution withParameters(IntegerTensor v) {
        return new StudentT(v);
    }

    private StudentT(IntegerTensor v) {
        this.v = v;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        DoubleTensor chi2Samples = ChiSquared.withParameters(v).sample(shape, random);
        return random.nextGaussian(shape).divInPlace(chi2Samples.divInPlace(v.toDouble()).sqrtInPlace());
    }

    @Override
    public DoubleTensor logProb(DoubleTensor t) {

        DoubleTensor vAsDouble = v.toDouble();
        DoubleTensor halfVPlusOne = vAsDouble.plus(1).divInPlace(2);

        DoubleTensor logGammaHalfVPlusOne = halfVPlusOne.logGamma();
        DoubleTensor logGammaHalfV = vAsDouble.div(2).logGammaInPlace();
        DoubleTensor halfLogV = vAsDouble.log().divInPlace(2);

        return logGammaHalfVPlusOne
            .minusInPlace(halfLogV)
            .minusInPlace(HALF_LOG_PI)
            .minusInPlace(logGammaHalfV)
            .minusInPlace(
                halfVPlusOne.timesInPlace(
                    t.pow(2).divInPlace(vAsDouble).plusInPlace(1).logInPlace()
                )
            );
    }

    @Override
    public Diffs dLogProb(DoubleTensor t) {
        DoubleTensor vAsDouble = v.toDouble();
        DoubleTensor dPdt = t.unaryMinus()
            .timesInPlace(vAsDouble.plus(1.0))
            .divInPlace(
                t.pow(2).plusInPlace(vAsDouble)
            );

        return new Diffs()
            .put(T, dPdt);
    }
}
