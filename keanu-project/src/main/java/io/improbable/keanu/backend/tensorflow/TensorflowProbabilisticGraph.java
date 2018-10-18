package io.improbable.keanu.backend.tensorflow;

import java.util.Map;

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
        DoubleTensor logProb = computableGraph.eval(inputs, logProbSumTotalOpName);
        return logProb.scalar();
    }

    @Override
    public void close() {
        computableGraph.close();
    }
}
