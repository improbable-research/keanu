package io.improbable.keanu.distributions.discrete;

import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.apache.commons.math3.util.CombinatoricsUtils;
import org.nd4j.linalg.util.ArrayUtil;

public class Binomial implements DiscreteDistribution {

    private final DoubleTensor successProbability;
    private final IntegerTensor numberOfTrials;

    /**
     * <h3>Binomial Distribution</h3>
     *
     * @param successProbability probability of success
     * @param numberOfTrials     number of trials
     * @see "Computer Generation of Statistical Distributions
     * by Richard Saucier
     * ARL-TR-2168 March 2000
     * 5.2.2 page 43"
     */
    public static DiscreteDistribution withParameters(DoubleTensor successProbability, IntegerTensor numberOfTrials) {
        return new Binomial(successProbability, numberOfTrials);
    }

    private Binomial(DoubleTensor successProbability, IntegerTensor numberOfTrials) {
        this.successProbability = successProbability;
        this.numberOfTrials = numberOfTrials;
    }

    @Override
    public IntegerTensor sample(int[] shape, KeanuRandom random) {
        Tensor.FlattenedView<Double> pWrapped = successProbability.getFlattenedView();
        Tensor.FlattenedView<Integer> nWrapped = numberOfTrials.getFlattenedView();

        int length = ArrayUtil.prod(shape);
        int[] samples = new int[length];
        for (int i = 0; i < length; i++) {
            samples[i] = sample(pWrapped.getOrScalar(i), nWrapped.getOrScalar(i), random);
        }

        return IntegerTensor.create(samples, shape);
    }

    private static int sample(double successProbability, int numberOfTrials, KeanuRandom random) {
        int sum = 0;
        for (int i = 0; i < numberOfTrials; i++) {
            if (random.nextDouble() < successProbability) {
                sum++;
            }
        }
        return sum;
    }

    @Override
    public DoubleTensor logProb(IntegerTensor x) {
        DoubleTensor logBinomialCoefficient = getLogBinomialCoefficient(x, numberOfTrials);

        DoubleTensor logBinomial = successProbability.pow(x.toDouble())
            .times(
                successProbability.unaryMinus().plusInPlace(1.0).powInPlace(numberOfTrials.minus(x).toDouble())
            ).logInPlace();

        return logBinomialCoefficient.plusInPlace(logBinomial);
    }

    private static DoubleTensor getLogBinomialCoefficient(IntegerTensor x, IntegerTensor numberOfTrials) {
        Tensor.FlattenedView<Integer> nWrapped = numberOfTrials.getFlattenedView();
        Tensor.FlattenedView<Integer> xWrapped = x.getFlattenedView();

        int length = (int) x.getLength();
        double[] logBinomialCoefficient = new double[length];
        for (int i = 0; i < length; i++) {
            logBinomialCoefficient[i] = getLogBinomialCoefficient(xWrapped.getOrScalar(i), nWrapped.getOrScalar(i));
        }

        return DoubleTensor.create(logBinomialCoefficient, x.getShape());
    }

    private static double getLogBinomialCoefficient(int x, int numberOfTrials) {
        long binomialCoefficient = CombinatoricsUtils.binomialCoefficient(numberOfTrials, x);
        return Math.log(binomialCoefficient);
    }

}