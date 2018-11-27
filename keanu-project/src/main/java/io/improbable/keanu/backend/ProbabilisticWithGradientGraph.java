package io.improbable.keanu.backend;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;

public interface ProbabilisticWithGradientGraph extends ProbabilisticGraph {

    Map<String, DoubleTensor> logProbGradients(Map<String, ?> inputs);

    Map<String, DoubleTensor> logProbGradients();

    Map<String, DoubleTensor> logLikelihoodGradients(Map<String, ?> inputs);

    Map<String, DoubleTensor> logLikelihoodGradients();

}
