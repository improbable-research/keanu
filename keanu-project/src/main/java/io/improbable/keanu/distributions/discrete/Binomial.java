package io.improbable.keanu.distributions.discrete;

import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.nd4j.linalg.util.ArrayUtil;

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

        Tensor.FlattenedView<Double> pWrapped = p.getFlattenedView();
        Tensor.FlattenedView<Integer> nWrapped = n.getFlattenedView();
        Tensor.FlattenedView<Integer> kWrapped = k.getFlattenedView();

        double[] logPmf = new double[(int) k.getLength()];
        for (int i = 0; i < logPmf.length; i++) {
            logPmf[i] = logPmf(kWrapped.getOrScalar(i), pWrapped.getOrScalar(i), nWrapped.getOrScalar(i));
        }

        return DoubleTensor.create(logPmf, k.getShape());
    }

    private static double logPmf(int k, double p, int n) {
        BinomialDistribution distribution = new BinomialDistribution(n, p);
        return distribution.logProbability(k);
    }
}
