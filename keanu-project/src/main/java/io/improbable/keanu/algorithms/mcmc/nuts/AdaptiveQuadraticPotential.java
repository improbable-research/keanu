package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.divide;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.dotProduct;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.pow;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.reciprocal;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.times;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.zeros;


public class AdaptiveQuadraticPotential implements Potential {

    private final int adaptCount;

    private Map<VariableReference, DoubleTensor> variance;
    private Map<VariableReference, DoubleTensor> standardDeviation;
    private Map<VariableReference, DoubleTensor> inverseStandardDeviation;

    private WeightedVariance forwardVariance;
    private WeightedVariance backgroundVariance;

    private final int adaptionWindow;

    private int nSamples;

    private KeanuRandom random;

    public AdaptiveQuadraticPotential(Map<VariableReference, DoubleTensor> initialMean,
                                      Map<VariableReference, DoubleTensor> initialVarianceDiagonal,
                                      double initialWeight,
                                      int adaptCount,
                                      KeanuRandom random) {

        this.adaptCount = adaptCount;
        this.variance = initialVarianceDiagonal;
        this.standardDeviation = pow(initialVarianceDiagonal, 0.5);
        this.inverseStandardDeviation = reciprocal(this.standardDeviation);
        this.forwardVariance = new WeightedVariance(initialMean, initialVarianceDiagonal, initialWeight);
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
        this.standardDeviation = pow(this.variance, 0.5);
        this.inverseStandardDeviation = reciprocal(this.standardDeviation);
    }

    @Override
    public Map<VariableReference, DoubleTensor> random() {

        Map<VariableReference, DoubleTensor> result = new HashMap<>();
        for (VariableReference variable : inverseStandardDeviation.keySet()) {
            DoubleTensor inverseStdForVariable = inverseStandardDeviation.get(variable);
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

        return 0.5 * dotProduct(momentum, velocity);
    }

    private static class WeightedVariance {

        private double weightSum;
        private Map<VariableReference, DoubleTensor> mean;
        private Map<VariableReference, DoubleTensor> rawVariance;

        WeightedVariance(Map<VariableReference, DoubleTensor> initialMean, Map<VariableReference, DoubleTensor> initialVariance, double initialWeight) {
            this.weightSum = initialWeight;
            this.mean = initialMean;
            this.rawVariance = times(initialVariance, weightSum);
        }

        private void addSample(Map<VariableReference, DoubleTensor> samples) {

            this.weightSum += 1.0;

            final double proportion = 1.0 / weightSum;

            for (VariableReference v : samples.keySet()) {

                final DoubleTensor oldMean = mean.get(v);
                final DoubleTensor sample = samples.get(v);

                final DoubleTensor oldDiff = sample.minus(oldMean);

                final DoubleTensor newMean = oldMean.plus(oldDiff.times(proportion));

                this.mean.put(v, newMean);

                DoubleTensor newDiff = sample.minus(newMean);

                DoubleTensor oldVariance = this.rawVariance.get(v);

                this.rawVariance.put(v, oldVariance.plus(oldDiff.times(newDiff)));
            }
        }

        private Map<VariableReference, DoubleTensor> currentVariance() {

            return divide(rawVariance, weightSum);
        }

    }
}
