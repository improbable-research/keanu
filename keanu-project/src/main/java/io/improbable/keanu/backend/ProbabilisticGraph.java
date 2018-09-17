package io.improbable.keanu.backend;

import java.util.List;
import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

public interface ProbabilisticGraph extends AutoCloseable {

    String LOG_PROB = "LOG_PROB__";

    double logProb(Map<String, DoubleTensor> inputs);

    Map<String, DoubleTensor> logProbGradients(Map<String, DoubleTensor> inputs);

    List<DoubleTensor> getOutputs(Map<String, DoubleTensor> inputs, List<String> outputs);

    @Override
    void close();
}
