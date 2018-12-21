package io.improbable.keanu.backend;

import lombok.Value;

import java.util.Map;

@Value
public class LogProbWithSample {

    private final double logProb;

    private final Map<VariableReference, ?> sample;

}
