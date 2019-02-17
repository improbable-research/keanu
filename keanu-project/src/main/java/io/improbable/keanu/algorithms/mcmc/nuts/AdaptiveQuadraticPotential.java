package io.improbable.keanu.algorithms.mcmc.nuts;

import com.google.common.base.Preconditions;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.Getter;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.dotProduct;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.pow;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.times;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.zeros;


public class AdaptiveQuadraticPotential implements Potential {

    private final long adaptCount;
    private final int adaptionWindowSize;
    private final KeanuRandom random;
    private VarianceCalculator forwardVariance;
    private VarianceCalculator backgroundVariance;
    private long nSamples;

    @Getter
    private Map<VariableReference, DoubleTensor> variance;

    @Getter
    private Map<VariableReference, DoubleTensor> standardDeviation;

    public AdaptiveQuadraticPotential(Map<VariableReference, DoubleTensor> initialMean,
                                      Map<VariableReference, DoubleTensor> initialVarianceDiagonal,
                                      double initialWeight,
                                      long adaptCount,
                                      int adaptionWindowSize,
                                      KeanuRandom random) {
        Preconditions.checkArgument(adaptionWindowSize > 1);

        this.adaptCount = adaptCount;
        this.setVariance(initialVarianceDiagonal);

        this.forwardVariance = new VarianceCalculator(initialMean, initialVarianceDiagonal, initialWeight);
        this.backgroundVariance = new VarianceCalculator(zeros(initialMean), zeros(initialMean), 0);
        this.adaptionWindowSize = adaptionWindowSize;
        this.random = random;
        this.nSamples = 0;
    }

    private void setVariance(Map<VariableReference, DoubleTensor> variance) {
        this.variance = variance;
        this.standardDeviation = pow(this.variance, 0.5);
    }

    @Override
    public void update(Map<VariableReference, DoubleTensor> position) {

        if (nSamples >= adaptCount) {
            return;
        }

        if (nSamples > 0 && nSamples % adaptionWindowSize == 0) {
            forwardVariance = backgroundVariance;
            backgroundVariance = new VarianceCalculator(zeros(variance), zeros(variance), 0);
        }

        forwardVariance.addSample(position);
        backgroundVariance.addSample(position);

        this.setVariance(forwardVariance.calculateCurrentVariance());

        nSamples++;
    }

    @Override
    public Map<VariableReference, DoubleTensor> randomMomentum() {

        Map<VariableReference, DoubleTensor> result = new HashMap<>();
        for (VariableReference variable : standardDeviation.keySet()) {

            DoubleTensor standardDeviationForVariable = standardDeviation.get(variable);

            DoubleTensor randomForVariable = random
                .nextGaussian(standardDeviationForVariable.getShape())
                .divInPlace(standardDeviationForVariable);

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
