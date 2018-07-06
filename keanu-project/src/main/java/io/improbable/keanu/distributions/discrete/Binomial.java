package io.improbable.keanu.distributions.discrete;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
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

        IntegerTensor binomialCoefficient = n.factorial().divInPlace(
            k.factorial().timesInPlace(n.minus(k).factorial())
        );

        DoubleTensor binomial = p.pow(k.toDouble())
            .times(
                p.unaryMinus().plus(1.0).pow(n.minus(k).toDouble())
            );

        return binomialCoefficient.toDouble().timesInPlace(binomial).logInPlace();
    }
}
