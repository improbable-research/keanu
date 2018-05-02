package io.improbable.keanu.vertices;

import java.util.Map;

public interface LogProbDifferentiable<T> {

    Map<String, Double> dLogProb(T value);

}
