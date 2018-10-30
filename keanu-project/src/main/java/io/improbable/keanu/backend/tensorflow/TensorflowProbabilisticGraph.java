package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.backend.LogProbWithSample;
import io.improbable.keanu.backend.ProbabilisticGraph;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.List;
import java.util.Map;

import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;

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
    public LogProbWithSample logProbWithSample(Map<String, ?> inputs, List<String> sampleFrom) {

        List<String> allOutputs = filterSampleFromForOutputs(inputs, sampleFrom);
        allOutputs.add(logProbSumTotalOpName);
        Map<String, Object> sample = takeSampleFromInputs(inputs, sampleFrom);

        Map<String, ?> results = computableGraph.compute(inputs, allOutputs);
        double logProb = ((DoubleTensor) results.get(logProbSumTotalOpName)).scalar();
        results.remove(logProbSumTotalOpName);

        sample.putAll(results);

        return new LogProbWithSample(logProb, sample);
    }

    private Map<String, Object> takeSampleFromInputs(Map<String, ?> inputs, List<String> sampleFrom) {
        return sampleFrom.stream()
            .filter(inputs::containsKey)
            .collect(toMap(output -> output, inputs::get));
    }

    private List<String> filterSampleFromForOutputs(Map<String, ?> inputs, List<String> sampleFrom) {
        return sampleFrom.stream()
            .filter(v -> !inputs.containsKey(v))
            .collect(toList());
    }

    @Override
    public void close() {
        computableGraph.close();
    }
}
