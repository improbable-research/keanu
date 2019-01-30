package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;

public interface FitnessFunctionGradient {

    Map<? extends VariableReference, DoubleTensor> value(Map<VariableReference, DoubleTensor> values);
}
