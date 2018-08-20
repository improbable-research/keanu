package io.improbable.keanu.distributions.discrete;

import io.improbable.keanu.distributions.Distribution;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Bernoulli implements Distribution<BooleanTensor> {

    private final DoubleTensor probTrue;

    public static Bernoulli withParameters(DoubleTensor probTrue) {
        return new Bernoulli(probTrue);
    }

    private Bernoulli(DoubleTensor probTrue) {
        this.probTrue = probTrue;
    }

    @Override
    public BooleanTensor sample(int[] shape, KeanuRandom random) {
        DoubleTensor uniforms = random.nextDouble(shape);
        return uniforms.lessThan(probTrue);
    }

    @Override
    public DoubleTensor logProb(BooleanTensor x) {
        DoubleTensor probTrueClamped = probTrue.clamp(DoubleTensor.ZERO_SCALAR, DoubleTensor.ONE_SCALAR);

        DoubleTensor probability = x.setDoubleIf(
            probTrueClamped,
            probTrueClamped.unaryMinus().plusInPlace(1.0)
        );

        return probability.logInPlace();
    }

    public DoubleTensor dLogProb(BooleanTensor x) {
        DoubleTensor greaterThanMask = probTrue
            .getGreaterThanMask(DoubleTensor.ONE_SCALAR);

        DoubleTensor lessThanOrEqualToMask = probTrue
            .getLessThanOrEqualToMask(DoubleTensor.ZERO_SCALAR);

        DoubleTensor greaterThanOneOrLessThanZero = greaterThanMask.plusInPlace(lessThanOrEqualToMask);

        DoubleTensor dlogProbdxForTrue = probTrue.reciprocal();
        dlogProbdxForTrue = dlogProbdxForTrue.setWithMaskInPlace(greaterThanOneOrLessThanZero, 0.0);

        DoubleTensor dlogProbdxForFalse = probTrue.minus(1.0).reciprocalInPlace();
        dlogProbdxForFalse = dlogProbdxForFalse.setWithMaskInPlace(greaterThanOneOrLessThanZero, 0.0);

        DoubleTensor dLogPdp = x.setDoubleIf(
            dlogProbdxForTrue,
            dlogProbdxForFalse
        );

        return dLogPdp;
    }
}
