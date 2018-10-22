package io.improbable.keanu.backend.tensorflow;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import io.improbable.keanu.backend.LogProbWithSample;
import io.improbable.keanu.backend.ProbabilisticGraph;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import lombok.Getter;

@AllArgsConstructor
public class TensorflowProbabilisticGraph implements ProbabilisticGraph {

    @Getter
    private final TensorflowComputableGraph computableGraph;

    @Getter
    private final String logProbSumTotalOpName;

    @Override
    public double logProb(Map<String, ?> inputs) {
        DoubleTensor logProb = computableGraph.compute(inputs, logProbSumTotalOpName);
        return logProb.scalar();
    }

    @Override
    public LogProbWithSample logProbWithSample(Map<String, ?> inputs, List<String> outputs) {
        List<String> allOutputs = new ArrayList<>(outputs);
        allOutputs.add(logProbSumTotalOpName);

        Map<String, ?> results = computableGraph.compute(inputs, allOutputs);
        double logProb = (Double) results.get(logProbSumTotalOpName);
        results.remove(logProbSumTotalOpName);

        return new LogProbWithSample(logProb, results);
    }

    @Override
    public void close() {
        computableGraph.close();
    }
}
