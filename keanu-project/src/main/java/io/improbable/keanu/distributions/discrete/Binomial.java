package io.improbable.keanu.distributions.discrete;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.apache.commons.math3.util.CombinatoricsUtils;
import org.nd4j.linalg.util.ArrayUtil;

public class Binomial {

    public static IntegerTensor sample(int[] shape, DoubleTensor p, IntegerTensor n, KeanuRandom random) {

        Tensor.FlattenedView<Double> pWrapped = p.getFlattenedView();
        Tensor.FlattenedView<Integer> nWrapped = n.getFlattenedView();

        int length = ArrayUtil.prod(shape);
        int[] samples = new int[length];
        for (int i = 0; i < length; i++) {
            samples[i] = sample(pWrapped.getOrScalar(i), nWrapped.getOrScalar(i), random);
        }

        return IntegerTensor.create(samples, shape);
    }

    public static int sample(double p, int n, KeanuRandom random) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            if (random.nextDouble() < p) {
                sum++;
            }
        }
        return sum;
    }

    public static DoubleTensor logPmf(IntegerTensor k, DoubleTensor p, IntegerTensor n) {

        DoubleTensor logBinomialCoefficient = getLogBinomialCoefficient(k, n);

        DoubleTensor logBinomial = p.pow(k.toDouble())
            .times(
                p.unaryMinus().plusInPlace(1.0).powInPlace(n.minus(k).toDouble())
            ).logInPlace();

        return logBinomialCoefficient.plusInPlace(logBinomial);
    }

    private static DoubleTensor getLogBinomialCoefficient(IntegerTensor k, IntegerTensor n) {
        Tensor.FlattenedView<Integer> nWrapped = n.getFlattenedView();
        Tensor.FlattenedView<Integer> kWrapped = k.getFlattenedView();

        int length = (int) k.getLength();
        double[] logBinomialCoefficient = new double[length];
        for (int i = 0; i < length; i++) {
            logBinomialCoefficient[i] = getLogBinomialCoefficient(kWrapped.getOrScalar(i), nWrapped.getOrScalar(i));
        }

        return DoubleTensor.create(logBinomialCoefficient, k.getShape());
    }

    private static double getLogBinomialCoefficient(int k, int n) {
        long binomialCoefficient = CombinatoricsUtils.binomialCoefficient(n, k);
        return Math.log(binomialCoefficient);
    }
}
