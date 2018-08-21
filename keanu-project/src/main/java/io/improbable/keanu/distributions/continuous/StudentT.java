package io.improbable.keanu.distributions.continuous;

import static java.lang.Math.PI;
import static java.lang.Math.log;

import static io.improbable.keanu.distributions.dual.Diffs.T;

import org.apache.commons.math3.special.Gamma;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

/**
 * @see "Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.1.23 page 36"
 */
public class StudentT implements ContinuousDistribution {

    private static final double HALF_LOG_PI = log(PI) / 2;
    private final IntegerTensor degreesOfFreedom;

    /**
     * @param degreesOfFreedom number of degrees of freedom
     * @return an instance of {@link ContinuousDistribution}
     */
    public static ContinuousDistribution withParameters(IntegerTensor degreesOfFreedom) {
        return new StudentT(degreesOfFreedom);
    }

    private StudentT(IntegerTensor degreesOfFreedom) {
        this.degreesOfFreedom = degreesOfFreedom;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        DoubleTensor chi2Samples = ChiSquared.withParameters(degreesOfFreedom).sample(shape, random);
        return random.nextGaussian(shape).divInPlace(chi2Samples.divInPlace(degreesOfFreedom.toDouble()).sqrtInPlace());
    }

    @Override
    public DoubleTensor logProb(DoubleTensor t) {

        DoubleTensor vAsDouble = degreesOfFreedom.toDouble();
        DoubleTensor halfVPlusOne = vAsDouble.plus(1).divInPlace(2);

        DoubleTensor logGammaHalfVPlusOne = halfVPlusOne.apply(Gamma::logGamma);
        DoubleTensor logGammaHalfV = vAsDouble.div(2).applyInPlace(Gamma::logGamma);
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
        DoubleTensor vAsDouble = degreesOfFreedom.toDouble();
        DoubleTensor dPdt = t.unaryMinus()
            .timesInPlace(vAsDouble.plus(1.0))
            .divInPlace(
                t.pow(2).plusInPlace(vAsDouble)
            );

        return new Diffs()
            .put(T, dPdt);
    }

}