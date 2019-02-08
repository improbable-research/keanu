package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.add;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.divide;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.pow;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.reciprocol;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.subtract;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.times;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.zeros;


public class QuadPotentialDiagAdapt implements Potential {

    private final int adaptCount;

    private Map<VariableReference, DoubleTensor> variance;
    private Map<VariableReference, DoubleTensor> stds;
    private Map<VariableReference, DoubleTensor> inverseStds;

    private WeightedVariance forwardVariance;
    private WeightedVariance backgroundVariance;

    private final int adaptionWindow;

    private int nSamples;

    private KeanuRandom random;


    public QuadPotentialDiagAdapt(Map<VariableReference, DoubleTensor> initialMean,
                                  Map<VariableReference, DoubleTensor> initialDiag,
                                  double initialWeight,
                                  int adaptCount,
                                  KeanuRandom random) {

        this.adaptCount = adaptCount;
        this.variance = initialDiag;
        this.stds = pow(initialDiag, 0.5);
        this.inverseStds = reciprocol(this.stds);
        this.forwardVariance = new WeightedVariance(initialMean, initialDiag, initialWeight);
        this.backgroundVariance = new WeightedVariance(zeros(initialMean), zeros(initialMean), 0);
        this.nSamples = 0;
        this.adaptionWindow = 101;
        this.random = random;
    }

    @Override
    public void update(Map<VariableReference, DoubleTensor> position, Map<? extends VariableReference, DoubleTensor> gradient, int sampleNum) {

        if (sampleNum > adaptCount) {
            return;
        }

        forwardVariance.addSample(position);
        backgroundVariance.addSample(position);

        updateFromWeightVar(forwardVariance);

        if (nSamples > 0 && nSamples % adaptionWindow == 0) {
            forwardVariance = backgroundVariance;
            backgroundVariance = new WeightedVariance(zeros(variance), zeros(variance), 0);
        }

        nSamples++;
    }

    private void updateFromWeightVar(WeightedVariance weightedVariance) {
        this.variance = weightedVariance.currentVariance();
        this.stds = pow(this.variance, 0.5);
        this.inverseStds = reciprocol(this.stds);
    }

    @Override
    public Map<VariableReference, DoubleTensor> random() {

        Map<VariableReference, DoubleTensor> result = new HashMap<>();
        for (VariableReference variable : inverseStds.keySet()) {
            DoubleTensor inverseStdForVariable = inverseStds.get(variable);
            DoubleTensor randomForVariable = random.nextGaussian(inverseStdForVariable.getShape()).timesInPlace(inverseStdForVariable);
            result.put(variable, randomForVariable);

        }

        return result;
    }

    @Override
    public Map<VariableReference, DoubleTensor> getVelocity(Map<VariableReference, DoubleTensor> momentum) {
        return times(variance, momentum);
    }

    @Override
    public double getKineticEnergy(Map<VariableReference, DoubleTensor> momentum,
                                   Map<VariableReference, DoubleTensor> velocity) {

        return 0.5 * VariableValues.dotProduct(momentum, velocity);
    }

    private static class WeightedVariance {

        private double weightSum;
        private Map<VariableReference, DoubleTensor> mean;
        private Map<VariableReference, DoubleTensor> rawVar;

        WeightedVariance(Map<VariableReference, DoubleTensor> initialMean, Map<VariableReference, DoubleTensor> initialVariance, double initialWeight) {
            this.weightSum = initialWeight;
            this.mean = initialMean;
            this.rawVar = times(initialVariance, weightSum);
        }

        private void addSample(Map<VariableReference, DoubleTensor> sample) {

            this.weightSum += 1.0;

            final double proportion = 1.0 / weightSum;

            final Map<VariableReference, DoubleTensor> oldDiff = subtract(sample, mean);
            this.mean = add(this.mean, times(oldDiff, proportion));

            final Map<VariableReference, DoubleTensor> newDiff = VariableValues.subtract(sample, mean);
            this.rawVar = add(this.rawVar, times(oldDiff, newDiff));
        }

        private Map<VariableReference, DoubleTensor> currentVariance() {

            return divide(rawVar, weightSum);
        }

    }
}
