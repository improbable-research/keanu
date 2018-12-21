package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.backend.Variable;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.backend.ProbabilisticWithGradientGraph;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TensorflowProbabilisticWithGradientGraph extends TensorflowProbabilisticGraph implements ProbabilisticWithGradientGraph {

    private final TensorflowComputableGraph computableGraph;
    private final Map<VariableReference, VariableReference> gradientOutputNameToInputName;

    public TensorflowProbabilisticWithGradientGraph(TensorflowComputableGraph computableGraph,
                                                    List<? extends Variable> latentVariables,
                                                    VariableReference logProbOp,
                                                    VariableReference logLikelihoodOp,
                                                    Map<VariableReference, VariableReference> gradientOutputNameToInputName) {
        super(computableGraph, latentVariables, logProbOp, logLikelihoodOp);
        this.computableGraph = computableGraph;
        this.gradientOutputNameToInputName = gradientOutputNameToInputName;
    }

    @Override
    public Map<? extends VariableReference, DoubleTensor> logProbGradients(Map<VariableReference, ?> inputs) {

        Map<VariableReference, ?> results = computableGraph.compute(inputs, gradientOutputNameToInputName.keySet());

        Map<VariableReference, DoubleTensor> gradientsByInputName = new HashMap<>();
        for (Map.Entry<VariableReference, ?> result : results.entrySet()) {
            gradientsByInputName.put(gradientOutputNameToInputName.get(result.getKey()), (DoubleTensor) result.getValue());
        }

        return gradientsByInputName;
    }

    @Override
    public Map<? extends VariableReference, DoubleTensor> logProbGradients() {
        return logProbGradients(null);
    }

    @Override
    public Map<? extends VariableReference, DoubleTensor> logLikelihoodGradients(Map<VariableReference, ?> inputs) {
        return null;
    }

    @Override
    public Map<? extends VariableReference, DoubleTensor> logLikelihoodGradients() {
        return null;
    }
}
