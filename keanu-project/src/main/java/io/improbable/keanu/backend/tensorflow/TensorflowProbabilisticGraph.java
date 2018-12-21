package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
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
    private final List<? extends Variable> latentVariables;

    @Getter
    private final VariableReference logProbOp;

    @Getter
    private final VariableReference logLikelihoodOp;

    @Override
    public double logProb(Map<VariableReference, ?> inputs) {
        DoubleTensor logProb = computableGraph.compute(inputs, logProbOp);
        return logProb.scalar();
    }

    @Override
    public double logLikelihood(Map<VariableReference, ?> inputs) {
        DoubleTensor logLikelihood = computableGraph.compute(inputs, logLikelihoodOp);
        return logLikelihood.scalar();
    }

    @Override
    public LogProbWithSample logProbWithSample(Map<VariableReference, ?> inputs, List<VariableReference> sampleFrom) {

        List<VariableReference> allOutputs = new ArrayList<>(sampleFrom);
        allOutputs.add(logProbOp);

        Map<VariableReference, ?> results = computableGraph.compute(inputs, allOutputs);
        double logProb = ((DoubleTensor) results.get(logProbOp)).scalar();
        results.remove(logProbOp);

        return new LogProbWithSample(logProb, results);
    }

    @Override
    public Map<VariableReference, ?> getLatentVariablesValues() {

        Map<VariableReference, ?> values = new HashMap<>();
        for (Variable latent : latentVariables) {
            values.put(latent.getReference(), computableGraph.getInput(latent.getReference()));
        }

        return values;
    }

    @Override
    public void close() {
        computableGraph.close();
    }
}
