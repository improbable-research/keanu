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

public class StudentT implements ContinuousDistribution {

    private static final double HALF_LOG_PI = log(PI) / 2;
    private final IntegerTensor alpha;

    /**
     * <h3>Student's T Distribution</h3>
     *
     * @param alpha shape parameter; number of degrees of freedom
     * @see "Computer Generation of Statistical Distributions
     * by Richard Saucier
     * ARL-TR-2168 March 2000
     * 5.1.23 page 36"
     */
    public static ContinuousDistribution withParameters(IntegerTensor alpha) {
        return new StudentT(alpha);
    }

    private StudentT(IntegerTensor alpha) {
        this.alpha = alpha;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        DoubleTensor chi2Samples = ChiSquared.withParameters(alpha).sample(shape, random);
        return random.nextGaussian(shape).divInPlace(chi2Samples.divInPlace(alpha.toDouble()).sqrtInPlace());
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {

        DoubleTensor dofAsDouble = alpha.toDouble();
        DoubleTensor halfDofPlusOne = dofAsDouble.plus(1).divInPlace(2);

        DoubleTensor logGammaHalfDofPlusOne = halfDofPlusOne.apply(Gamma::logGamma);
        DoubleTensor logGammaHalfDof = dofAsDouble.div(2).applyInPlace(Gamma::logGamma);
        DoubleTensor halfLogDof = dofAsDouble.log().divInPlace(2);

        return logGammaHalfDofPlusOne
            .minusInPlace(halfLogDof)
            .minusInPlace(HALF_LOG_PI)
            .minusInPlace(logGammaHalfDof)
            .minusInPlace(
                halfDofPlusOne.timesInPlace(
                    x.pow(2).divInPlace(dofAsDouble).plusInPlace(1).logInPlace()
                )
            );
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        DoubleTensor dofAsDouble = alpha.toDouble();
        DoubleTensor dPdx = x.unaryMinus()
            .timesInPlace(dofAsDouble.plus(1.0))
            .divInPlace(
                x.pow(2).plusInPlace(dofAsDouble)
            );

        return new Diffs()
            .put(T, dPdx);
    }

}