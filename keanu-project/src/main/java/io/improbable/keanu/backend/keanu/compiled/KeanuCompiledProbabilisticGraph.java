package io.improbable.keanu.backend.keanu.compiled;

import io.improbable.keanu.algorithms.ProbabilisticModel;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.backend.VariableImpl;
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
public class KeanuCompiledProbabilisticGraph implements ProbabilisticModel {

    /**
     * Takes a BayesianNetwork and converts it to a KeanuCompiledProbabilisticGraph. This compiles the graph in the
     * bayesian network and then compiles the logProb graph.
     *
     * @param network The bayesian network for conversions
     * @return A compiled ProbabilisticGraph that represents the BayesianNetwork
     */
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

        List latentVariables = builder.getLatentVariables().stream()
            .map(v -> new VariableImpl<>(computableGraph, v))
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
    private final List<Variable> latentVariables;

    @Getter
    private final VariableReference logProbOp;

    @Getter
    private final VariableReference logLikelihoodOp;

    @Override
    public double logProb(Map<VariableReference, ?> inputs) {
        DoubleTensor logProb = (DoubleTensor) computableGraph.compute(inputs).get(logProbOp);
        return logProb.scalar();
    }

    @Override
    public double logLikelihood(Map<VariableReference, ?> inputs) {

        if (logLikelihoodOp == null) {
            throw new IllegalStateException("Likelihood is undefined");
        }

        DoubleTensor logLikelihood = (DoubleTensor) computableGraph.compute(inputs).get(logLikelihoodOp);
        return logLikelihood.scalar();
    }

}
