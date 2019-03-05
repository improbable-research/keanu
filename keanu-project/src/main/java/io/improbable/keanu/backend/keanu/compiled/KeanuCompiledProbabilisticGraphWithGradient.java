package io.improbable.keanu.backend.keanu.compiled;

import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.backend.VariableImpl;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import static io.improbable.keanu.backend.ProbabilisticGraphConverter.convertLogProbObservation;
import static io.improbable.keanu.backend.ProbabilisticGraphConverter.convertLogProbPrior;

public class KeanuCompiledProbabilisticGraphWithGradient extends KeanuCompiledProbabilisticGraph implements ProbabilisticModelWithGradient {

    public static KeanuCompiledProbabilisticGraphWithGradient convert(BayesianNetwork network) {

        KeanuCompiledGraphWithGradientBuilder builder = new KeanuCompiledGraphWithGradientBuilder();

        builder.convert(network.getVertices());

        Optional<VariableReference> logLikelihoodReference = convertLogProbObservation(network, builder);
        VariableReference priorLogProbReference = convertLogProbPrior(network, builder);

        VariableReference logProbReference = logLikelihoodReference
            .map(ll -> builder.add(ll, priorLogProbReference))
            .orElse(priorLogProbReference);

        List<VariableReference> latentVariablesReferences = network.getLatentVertices().stream()
            .map(Variable::getReference)
            .collect(Collectors.toList());


        Optional<Map<VariableReference, VariableReference>> logLikelihoodGradients = logLikelihoodReference.map(
            logLikelihoodOp -> builder.addGradients(
                logLikelihoodOp,
                latentVariablesReferences
            )
        );

        Map<VariableReference, VariableReference> logProbGradients = builder.addGradients(
            logProbReference,
            latentVariablesReferences
        );

        WrappedCompiledGraph computableGraph = builder.build();

        List latentVariables = builder.getLatentVariables().stream()
            .map(v -> new VariableImpl<>(computableGraph, v))
            .collect(Collectors.toList());

        return new KeanuCompiledProbabilisticGraphWithGradient(
            computableGraph,
            latentVariables,
            logProbReference,
            logLikelihoodReference.orElse(null),
            logProbGradients,
            logLikelihoodGradients.orElse(null)
        );

    }

    private final WrappedCompiledGraph computableGraph;
    private final Map<VariableReference, VariableReference> logProbGradients;
    private final Map<VariableReference, VariableReference> logLikelihoodGradients;

    public KeanuCompiledProbabilisticGraphWithGradient(WrappedCompiledGraph computableGraph,
                                                       List<Variable> latentVariables,
                                                       VariableReference logProbOp,
                                                       VariableReference logLikelihoodOp,
                                                       Map<VariableReference, VariableReference> logProbGradients,
                                                       Map<VariableReference, VariableReference> logLikelihoodGradients) {
        super(computableGraph, latentVariables, logProbOp, logLikelihoodOp);
        this.computableGraph = computableGraph;
        this.logProbGradients = logProbGradients;
        this.logLikelihoodGradients = logLikelihoodGradients;
    }

    @Override
    public Map<VariableReference, DoubleTensor> logProbGradients() {
        return logProbGradients(null);
    }

    @Override
    public Map<VariableReference, DoubleTensor> logProbGradients(Map<VariableReference, ?> inputs) {
        return calculateGradients(inputs, logProbGradients);
    }

    @Override
    public Map<VariableReference, DoubleTensor> logLikelihoodGradients() {
        return logLikelihoodGradients(null);
    }

    @Override
    public Map<VariableReference, DoubleTensor> logLikelihoodGradients(Map<VariableReference, ?> inputs) {
        return calculateGradients(inputs, logLikelihoodGradients);
    }

    private Map<VariableReference, DoubleTensor> calculateGradients(Map<VariableReference, ?> inputs,
                                                                    Map<VariableReference, VariableReference> gradientLookup) {

        Map<VariableReference, ?> results = computableGraph.compute(inputs, gradientLookup.keySet());

        Map<VariableReference, DoubleTensor> gradientsByInputName = new HashMap<>();
        for (Map.Entry<VariableReference, ?> result : results.entrySet()) {
            gradientsByInputName.put(gradientLookup.get(result.getKey()), (DoubleTensor) result.getValue());
        }

        return gradientsByInputName;
    }
}

