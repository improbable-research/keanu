package io.improbable.keanu.algorithms.mcmc.nuts;

import com.google.common.base.Preconditions;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;

import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.add;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.divide;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.subtract;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.times;

/**
 * Uses Welford's online algorithm for computing sample variance
 */
public class VarianceCalculator {

    private double count;
    private Map<VariableReference, DoubleTensor> mean;
    private Map<VariableReference, DoubleTensor> M2;

    public VarianceCalculator(Map<VariableReference, DoubleTensor> initialMean, Map<VariableReference, DoubleTensor> initialVariance, double initialWeight) {
        Preconditions.checkArgument(initialWeight >= 0.0);
        this.count = initialWeight;
        this.mean = initialMean;
        this.M2 = times(initialVariance, count);
    }

    public void addSample(Map<VariableReference, DoubleTensor> sample) {

        this.count += 1.0;

        final Map<VariableReference, DoubleTensor> delta = subtract(sample, mean);

        this.mean = add(this.mean, divide(delta, count));

        final Map<VariableReference, DoubleTensor> delta2 = subtract(sample, mean);

        this.M2 = add(this.M2, times(delta, delta2));
    }

    public Map<VariableReference, DoubleTensor> currentVariance() {
        return divide(M2, count);
    }

}
