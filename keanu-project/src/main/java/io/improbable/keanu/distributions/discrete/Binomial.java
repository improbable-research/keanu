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

        Tensor.FlattenedView<Double> pWrapped = p.getFlattenedView();
        Tensor.FlattenedView<Integer> nWrapped = n.getFlattenedView();
        Tensor.FlattenedView<Integer> kWrapped = k.getFlattenedView();

        int length = (int) k.getLength();
        double[] pmf = new double[length];
        for (int i = 0; i < length; i++) {
            pmf[i] = logPmf(kWrapped.getOrScalar(i), pWrapped.getOrScalar(i), nWrapped.getOrScalar(i));
        }

        return DoubleTensor.create(pmf, k.getShape());
    }

    private static double logPmf(int k, double p, int n) {
        long binomialCoefficient = CombinatoricsUtils.binomialCoefficient(n, k);
        double binomial = Math.pow(p, k) * Math.pow(1.0 - p, n - k);
        return Math.log(binomialCoefficient * binomial);
    }
}
