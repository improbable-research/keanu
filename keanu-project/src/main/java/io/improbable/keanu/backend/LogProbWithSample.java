package io.improbable.keanu.backend;

import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import lombok.Value;

import java.util.Map;

@Value
public class LogProbWithSample {

    private final double logProb;

    private final Map<VariableReference, ?> sample;

}
