package io.improbable.keanu.distributions.discrete;

import io.improbable.keanu.distributions.Distribution;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

/**
 * @see "Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.2.1 page 42"
 */
public class Bernoulli implements Distribution<BooleanTensor> {

    private final DoubleTensor probOfEvent;

    /**
     * @param probOfEvent probability of an event
     * @return an instance of {@link Bernoulli}
     */
    public static Bernoulli withParameters(DoubleTensor probOfEvent) {
        return new Bernoulli(probOfEvent);
    }

    private Bernoulli(DoubleTensor probOfEvent) {
        this.probOfEvent = probOfEvent;
    }

    @Override
    public BooleanTensor sample(int[] shape, KeanuRandom random) {
        DoubleTensor uniforms = random.nextDouble(shape);
        return uniforms.lessThan(probOfEvent);
    }

    @Override
    public DoubleTensor logProb(BooleanTensor x) {
        DoubleTensor probTrueClamped = probOfEvent.clamp(DoubleTensor.ZERO_SCALAR, DoubleTensor.ONE_SCALAR);

        DoubleTensor probability = x.setDoubleIf(
            probTrueClamped,
            probTrueClamped.unaryMinus().plusInPlace(1.0)
        );

        return probability.logInPlace();
    }

    public DoubleTensor dLogProb(BooleanTensor x) {
        DoubleTensor greaterThanMask = probOfEvent
            .getGreaterThanMask(DoubleTensor.ONE_SCALAR);

        DoubleTensor lessThanOrEqualToMask = probOfEvent
            .getLessThanOrEqualToMask(DoubleTensor.ZERO_SCALAR);

        DoubleTensor greaterThanOneOrLessThanZero = greaterThanMask.plusInPlace(lessThanOrEqualToMask);

        DoubleTensor dlogProbdxForTrue = probOfEvent.reciprocal();
        dlogProbdxForTrue = dlogProbdxForTrue.setWithMaskInPlace(greaterThanOneOrLessThanZero, 0.0);

        DoubleTensor dlogProbdxForFalse = probOfEvent.minus(1.0).reciprocalInPlace();
        dlogProbdxForFalse = dlogProbdxForFalse.setWithMaskInPlace(greaterThanOneOrLessThanZero, 0.0);

        DoubleTensor dLogPdp = x.setDoubleIf(
            dlogProbdxForTrue,
            dlogProbdxForFalse
        );

        return dLogPdp;
    }
}
