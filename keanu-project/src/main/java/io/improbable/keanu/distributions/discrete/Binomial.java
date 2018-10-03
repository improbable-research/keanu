package io.improbable.keanu.distributions.discrete;

import org.nd4j.linalg.util.ArrayUtil;

import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Binomial implements DiscreteDistribution {

    private final DoubleTensor p;
    private final IntegerTensor n;

    public static DiscreteDistribution withParameters(DoubleTensor p, IntegerTensor n) {
        return new Binomial(p, n);
    }

    private Binomial(DoubleTensor p, IntegerTensor n) {
        this.p = p;
        this.n = n;
    }

    @Override
    public IntegerTensor sample(int[] shape, KeanuRandom random) {
        Tensor.FlattenedView<Double> pWrapped = p.getFlattenedView();
        Tensor.FlattenedView<Integer> nWrapped = n.getFlattenedView();

        int length = ArrayUtil.prod(shape);
        int[] samples = new int[length];
        for (int i = 0; i < length; i++) {
            samples[i] = sample(pWrapped.getOrScalar(i), nWrapped.getOrScalar(i), random);
        }

        return IntegerTensor.create(samples, shape);
    }

    private static int sample(double p, int n, KeanuRandom random) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            if (random.nextDouble() < p) {
                sum++;
            }
        }
        return sum;
    }

    @Override
    public DoubleTensor logProb(IntegerTensor k) {
        DoubleTensor logBinomialCoefficient = getLogBinomialCoefficient(k, n);

        DoubleTensor kDouble = k.toDouble();
        DoubleTensor nDouble = n.toDouble();
        DoubleTensor kLogP = kDouble.times(p.log());
        DoubleTensor logOneMinusP = p.unaryMinus().plusInPlace(1.0).logInPlace();
        DoubleTensor nMinusKLogOneMinusP = nDouble.minusInPlace(kDouble).timesInPlace(logOneMinusP);

        return logBinomialCoefficient.plusInPlace(kLogP).plusInPlace(nMinusKLogOneMinusP);
    }

    private static DoubleTensor getLogBinomialCoefficient(IntegerTensor k, IntegerTensor n) {

        DoubleTensor nDouble = n.toDouble();
        DoubleTensor kDouble = k.toDouble();
        DoubleTensor logNFactorial = nDouble.plus(1.0).logGammaInPlace();
        DoubleTensor logKFactorial = kDouble.plus(1.0).logGammaInPlace();
        DoubleTensor logNMinusKFactorial = nDouble.minusInPlace(kDouble).plusInPlace(1.0).logGammaInPlace();

        return logNFactorial.minusInPlace(logKFactorial).minusInPlace(logNMinusKFactorial);
    }
}
