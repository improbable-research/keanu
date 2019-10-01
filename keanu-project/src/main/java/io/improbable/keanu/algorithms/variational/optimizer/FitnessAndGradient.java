package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.Value;

import java.util.Map;

@Value
public class FitnessAndGradient {

    double fitness;
    Map<VariableReference, DoubleTensor> gradients;
}
