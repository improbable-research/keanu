package io.improbable.keanu.distributions.discrete;

import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.apache.commons.math3.util.CombinatoricsUtils;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * @see "Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.2.2 page 43"
 */
public class Binomial implements DiscreteDistribution {

    private final IntegerTensor numberOfTrials;
    private final DoubleTensor successProbability;

    /**
     * @param numberOfTrials     number of trials
     * @param successProbability probability of success
     * @return an instance of {@link DiscreteDistribution}
     */
    public static DiscreteDistribution withParameters(IntegerTensor numberOfTrials, DoubleTensor successProbability) {
        return new Binomial(numberOfTrials, successProbability);
    }

    private Binomial(IntegerTensor numberOfTrials, DoubleTensor successProbability) {
        this.numberOfTrials = numberOfTrials;
        this.successProbability = successProbability;
    }

    @Override
    public IntegerTensor sample(int[] shape, KeanuRandom random) {
        Tensor.FlattenedView<Integer> nWrapped = numberOfTrials.getFlattenedView();
        Tensor.FlattenedView<Double> pWrapped = successProbability.getFlattenedView();

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