package io.improbable.keanu.distributions.discrete;

import io.improbable.keanu.distributions.Distribution;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Bernoulli implements Distribution<BooleanTensor> {

    private final DoubleTensor probTrue;

    public static Distribution<BooleanTensor> withParameters(DoubleTensor probTrue) {
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
}
