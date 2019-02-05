package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;

public interface FitnessFunction {

    /**
     * @param values the values for each variable in the function
     * @return the fitness at the values specified
     */
    double getFitnessAt(Map<VariableReference, DoubleTensor> values);
}
