package io.improbable.keanu.algorithms.mcmc.nuts;

import com.google.common.base.Preconditions;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;

import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.divide;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.times;

/**
 * Uses Welford's online algorithm for computing sample variance
 */
public class VarianceCalculator {

    private double count;
    private Map<VariableReference, DoubleTensor> mean;
    private Map<VariableReference, DoubleTensor> M2;

    public VarianceCalculator(Map<VariableReference, DoubleTensor> initialMean,
                              Map<VariableReference, DoubleTensor> initialVariance,
                              double initialWeight) {
        Preconditions.checkArgument(initialWeight >= 0.0);
        this.count = initialWeight;
        this.mean = initialMean;
        this.M2 = times(initialVariance, count);
    }

    public void addSample(Map<VariableReference, DoubleTensor> sampleForLatents) {

        this.count += 1.0;

        for (Map.Entry<VariableReference, DoubleTensor> sampleForVariable : sampleForLatents.entrySet()) {

            final VariableReference v = sampleForVariable.getKey();
            final DoubleTensor sample = sampleForVariable.getValue();

            final DoubleTensor oldMean = mean.get(v);

            final DoubleTensor delta = sample.minus(oldMean);

            final DoubleTensor newMean = oldMean.plus(delta.div(count));

            final DoubleTensor delta2 = sample.minus(newMean);

            final DoubleTensor oldM2 = this.M2.get(v);
            final DoubleTensor newM2 = oldM2.plus(delta.times(delta2));

            mean.put(v, newMean);
            M2.put(v, newM2);
        }
    }

    public Map<VariableReference, DoubleTensor> calculateCurrentVariance() {
        return divide(M2, count);
    }

}
