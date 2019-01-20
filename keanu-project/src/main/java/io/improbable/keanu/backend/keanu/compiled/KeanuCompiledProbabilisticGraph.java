package io.improbable.keanu.backend.keanu.compiled;

import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.backend.ProbabilisticGraph;
import io.improbable.keanu.backend.Variable;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.backend.tensorflow.TensorflowVariable;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import static io.improbable.keanu.backend.ProbabilisticGraphConverter.convertLogProbObservation;
import static io.improbable.keanu.backend.ProbabilisticGraphConverter.convertLogProbPrior;

@AllArgsConstructor
public class KeanuCompiledProbabilisticGraph implements ProbabilisticGraph {

    public static KeanuCompiledProbabilisticGraph convert(BayesianNetwork network) {
        KeanuCompiledGraphBuilder builder = new KeanuCompiledGraphBuilder();

        builder.convert(network.getVertices());

        Optional<VariableReference> logLikelihoodReference = convertLogProbObservation(network, builder);
        VariableReference priorLogProbReference = convertLogProbPrior(network, builder);

        VariableReference logProbReference = logLikelihoodReference
            .map(ll -> builder.add(ll, priorLogProbReference))
            .orElse(priorLogProbReference);

        builder.registerOutput(logProbReference);
        logLikelihoodReference.ifPresent(builder::registerOutput);

        ComputableGraph computableGraph = builder.build();

        List<Variable<?>> latentVariables = builder.getLatentVariables().stream()
            .map(v -> new TensorflowVariable<>(computableGraph, v))
            .collect(Collectors.toList());

        return new KeanuCompiledProbabilisticGraph(
            computableGraph,
            latentVariables,
            logProbReference,
            logLikelihoodReference.orElse(null)
        );
    }

    @Getter
    private final ComputableGraph computableGraph;

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
}
