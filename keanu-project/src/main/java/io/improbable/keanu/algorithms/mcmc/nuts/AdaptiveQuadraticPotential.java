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
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.withShape;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.zeros;


public class AdaptiveQuadraticPotential implements Potential {

    private final double initialWeight;
    private final double initialMean;
    private final double initialVariance;
    private final int adaptionWindowSize;
    private VarianceCalculator forwardVariance;
    private VarianceCalculator backgroundVariance;
    private long nSamples;

    @Getter
    private Map<VariableReference, DoubleTensor> variance;

    @Getter
    private Map<VariableReference, DoubleTensor> standardDeviation;

    public AdaptiveQuadraticPotential(double initialMean,
                                      double initialVariance,
                                      double initialWeight,
                                      int adaptionWindowSize) {
        Preconditions.checkArgument(adaptionWindowSize > 1);

        this.initialWeight = initialWeight;
        this.initialMean = initialMean;
        this.initialVariance = initialVariance;

        this.adaptionWindowSize = adaptionWindowSize;
        this.nSamples = 0;
    }

    public void initialize(Map<VariableReference, DoubleTensor> shapeLike) {

        Map<VariableReference, DoubleTensor> varianceShapedLike = withShape(initialVariance, shapeLike);
        Map<VariableReference, DoubleTensor> meanShapedLike = withShape(initialMean, shapeLike);

        this.setVariance(varianceShapedLike);

        this.forwardVariance = new VarianceCalculator(meanShapedLike, varianceShapedLike, initialWeight);
        this.backgroundVariance = new VarianceCalculator(zeros(meanShapedLike), zeros(meanShapedLike), 0);
    }

    private void setVariance(Map<VariableReference, DoubleTensor> variance) {
        this.variance = variance;
        this.standardDeviation = pow(this.variance, 0.5);
    }

    @Override
    public void update(Map<VariableReference, DoubleTensor> position) {

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
    public Map<VariableReference, DoubleTensor> randomMomentum(KeanuRandom random) {

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
