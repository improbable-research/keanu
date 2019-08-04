package io.improbable.keanu.distributions.discrete;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.Distribution;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.tensor.If;
import io.improbable.keanu.vertices.tensor.bool.BooleanPlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;

public class Bernoulli implements Distribution<BooleanTensor> {

    private final DoubleTensor probTrue;

    public static Bernoulli withParameters(DoubleTensor probTrue) {
        return new Bernoulli(probTrue);
    }

    private Bernoulli(DoubleTensor probTrue) {
        this.probTrue = probTrue;
    }

    @Override
    public BooleanTensor sample(long[] shape, KeanuRandom random) {
        DoubleTensor uniforms = random.nextDouble(shape);
        return uniforms.lessThan(probTrue);
    }

    @Override
    public DoubleTensor logProb(BooleanTensor x) {
        DoubleTensor probTrueClamped = probTrue.clamp(DoubleTensor.scalar(0.0), DoubleTensor.scalar(1.0));

        DoubleTensor probability = probTrueClamped.where(x, probTrueClamped.unaryMinus().plusInPlace(1.0));

        return probability.logInPlace();
    }

    public static DoubleVertex logProbGraph(BooleanPlaceholderVertex x, DoublePlaceholderVertex probTrue) {
        DoubleVertex zero = ConstantVertex.of(DoubleTensor.zeros(x.getShape()));
        DoubleVertex one = ConstantVertex.of(DoubleTensor.ones(x.getShape()));
        DoubleVertex probTrueClamped = DoubleVertex.min(DoubleVertex.max(probTrue, zero), one);

        DoubleVertex probability = If.isTrue(x)
            .then(probTrueClamped)
            .orElse(probTrueClamped.unaryMinus().plus(1.));

        return probability.log();
    }

    public DoubleTensor dLogProb(BooleanTensor x) {
        DoubleTensor greaterThanMask = probTrue
            .greaterThanMask(DoubleTensor.scalar(1.0));

        DoubleTensor lessThanOrEqualToMask = probTrue
            .lessThanOrEqualToMask(DoubleTensor.scalar(0.0));

        DoubleTensor greaterThanOneOrLessThanZero = greaterThanMask.plusInPlace(lessThanOrEqualToMask);

        DoubleTensor dlogProbdxForTrue = probTrue.reciprocal();
        dlogProbdxForTrue = dlogProbdxForTrue.setWithMaskInPlace(greaterThanOneOrLessThanZero, 0.0);

        DoubleTensor dlogProbdxForFalse = probTrue.minus(1.0).reciprocalInPlace();
        dlogProbdxForFalse = dlogProbdxForFalse.setWithMaskInPlace(greaterThanOneOrLessThanZero, 0.0);

        DoubleTensor dLogPdp =dlogProbdxForTrue.where(x, dlogProbdxForFalse);

        return dLogPdp;
    }
}
