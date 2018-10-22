package io.improbable.keanu.backend;

import java.util.Map;

import lombok.Value;

@Value
public class LogProbWithSample {

    private final double logProb;

    private final Map<String, ?> sample;

}
