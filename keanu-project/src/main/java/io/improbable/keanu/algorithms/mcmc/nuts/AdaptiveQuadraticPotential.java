package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.dotProduct;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.times;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.zeros;


public class AdaptiveQuadraticPotential implements Potential {

    private final int adaptCount;
    private final int adaptionWindowSize;
    private final KeanuRandom random;

    private Map<VariableReference, DoubleTensor> variance;

    private VarianceCalculator forwardVariance;
    private VarianceCalculator backgroundVariance;

    private long nSamples;

    public AdaptiveQuadraticPotential(Map<VariableReference, DoubleTensor> initialMean,
                                      Map<VariableReference, DoubleTensor> initialVarianceDiagonal,
                                      double initialWeight,
                                      int adaptCount,
                                      int adaptionWindowSize,
                                      KeanuRandom random) {

        this.adaptCount = adaptCount;
        this.variance = initialVarianceDiagonal;

        this.forwardVariance = new VarianceCalculator(initialMean, initialVarianceDiagonal, initialWeight);
        this.backgroundVariance = new VarianceCalculator(zeros(initialMean), zeros(initialMean), 0);
        this.adaptionWindowSize = adaptionWindowSize;
        this.random = random;
        this.nSamples = 0;
    }

    @Override
    public void update(Map<VariableReference, DoubleTensor> position) {

        if (nSamples >= adaptCount) {
            return;
        }

        forwardVariance.addSample(position);
        backgroundVariance.addSample(position);

        this.variance = forwardVariance.currentVariance();

        if (nSamples > 0 && nSamples % adaptionWindowSize == 0) {
            forwardVariance = backgroundVariance;
            backgroundVariance = new VarianceCalculator(zeros(variance), zeros(variance), 0);
        }

        nSamples++;
    }

    @Override
    public Map<VariableReference, DoubleTensor> random() {

        Map<VariableReference, DoubleTensor> result = new HashMap<>();
        for (VariableReference variable : variance.keySet()) {

            DoubleTensor varianceForVariable = variance.get(variable);

            DoubleTensor randomForVariable = random
                .nextGaussian(varianceForVariable.getShape())
                .divInPlace(varianceForVariable.pow(0.5));

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

}
