package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.backend.ProbabilisticGraph;
import io.improbable.keanu.backend.Variable;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.List;
import java.util.Map;


@AllArgsConstructor
public class TensorflowProbabilisticGraph implements ProbabilisticGraph {

    public static TensorflowProbabilisticGraph convert(BayesianNetwork network) {
        TensorflowProbabilisticGraphBuilder builder = new TensorflowProbabilisticGraphBuilder();
        builder.convert(network);

        return builder.build();
    }

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

        if (logLikelihoodOp == null) {
            throw new IllegalStateException("Likelihood is undefined");
        }

        DoubleTensor logLikelihood = computableGraph.compute(inputs, logLikelihoodOp);
        return logLikelihood.scalar();
    }

    @Override
    public void close() {
        computableGraph.close();
    }
}
