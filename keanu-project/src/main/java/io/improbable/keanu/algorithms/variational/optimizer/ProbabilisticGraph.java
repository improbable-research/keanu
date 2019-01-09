package io.improbable.keanu.algorithms.variational.optimizer;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

public interface ProbabilisticGraph {

    default double logProb() {
        return logProb(Collections.emptyList());
    }

    double logProb(List<? extends Variable> inputs);

    double logProbOfProbabilisticVertices();

    default double logLikelihood() {
        return logLikelihood(Collections.emptyList());
    }

    double logLikelihood(List<? extends Variable> inputs);

    List<? extends Variable> getLatentVariables();

    List<? extends Variable<DoubleTensor>> getContinuousLatentVariables();

}
