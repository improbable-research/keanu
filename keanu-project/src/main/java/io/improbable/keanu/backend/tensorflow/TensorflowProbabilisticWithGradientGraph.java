package io.improbable.keanu.backend.tensorflow;

import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.backend.ProbabilisticWithGradientGraph;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

public class TensorflowProbabilisticWithGradientGraph extends TensorflowProbabilisticGraph implements ProbabilisticWithGradientGraph {

    private final TensorflowComputableGraph computableGraph;
    private final Map<String, String> gradientOutputNameToInputName;

    public TensorflowProbabilisticWithGradientGraph(TensorflowComputableGraph computableGraph,
                                                    String logProbSumTotalOpName,
                                                    Map<String, String> gradientOutputNameToInputName) {
        super(computableGraph, logProbSumTotalOpName);
        this.computableGraph = computableGraph;
        this.gradientOutputNameToInputName = gradientOutputNameToInputName;

    }

    @Override
    public Map<String, DoubleTensor> logProbGradients(Map<String, ?> inputs) {

        Map<String, ?> results = computableGraph.compute(inputs, gradientOutputNameToInputName.keySet());

        Map<String, DoubleTensor> gradientsByInputName = new HashMap<>();
        for (Map.Entry<String, ?> result : results.entrySet()) {
            gradientsByInputName.put(gradientOutputNameToInputName.get(result.getKey()), (DoubleTensor) result.getValue());
        }

        return gradientsByInputName;
    }
}
