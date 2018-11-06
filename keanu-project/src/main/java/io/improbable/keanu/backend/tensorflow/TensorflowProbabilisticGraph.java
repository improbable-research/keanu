package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.backend.LogProbWithSample;
import io.improbable.keanu.backend.ProbabilisticGraph;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


@AllArgsConstructor
public class TensorflowProbabilisticGraph implements ProbabilisticGraph {

    @Getter
    private final TensorflowComputableGraph computableGraph;

    @Getter
    private final List<String> latentVariables;

    @Getter
    private final String logProbSumTotalOpName;

    @Override
    public double logProb(Map<String, ?> inputs) {
        DoubleTensor logProb = computableGraph.compute(inputs, logProbSumTotalOpName);
        return logProb.scalar();
    }

    @Override
    public LogProbWithSample logProbWithSample(Map<String, ?> inputs, List<String> sampleFrom) {

        List<String> allOutputs = new ArrayList<>(sampleFrom);
        allOutputs.add(logProbSumTotalOpName);

        Map<String, ?> results = computableGraph.compute(inputs, allOutputs);
        double logProb = ((DoubleTensor) results.get(logProbSumTotalOpName)).scalar();
        results.remove(logProbSumTotalOpName);

        return new LogProbWithSample(logProb, results);
    }

    @Override
    public Map<String, ?> getLatentVariablesValues() {

        Map<String, ?> values = new HashMap<>();
        for (String latent : latentVariables) {
            values.put(latent, computableGraph.getInput(latent));
        }

        return values;
    }

    @Override
    public void close() {
        computableGraph.close();
    }
}
