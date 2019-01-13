package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.backend.ProbabilisticGraphWithGradient;
import io.improbable.keanu.backend.Variable;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import static io.improbable.keanu.backend.ProbabilisticGraphConverter.convertLogProbObservation;
import static io.improbable.keanu.backend.ProbabilisticGraphConverter.convertLogProbPrior;

public class TensorflowProbabilisticGraphWithGradient extends TensorflowProbabilisticGraph implements ProbabilisticGraphWithGradient {

    public static TensorflowProbabilisticGraphWithGradient convert(BayesianNetwork network) {

        TensorflowComputableGraphBuilder builder = new TensorflowComputableGraphBuilder();

        builder.convert(network.getVertices());

        Optional<VariableReference> logLikelihoodReference = convertLogProbObservation(network, builder);
        VariableReference priorLogProbReference = convertLogProbPrior(network, builder);

        VariableReference logProbReference = logLikelihoodReference
            .map(ll -> builder.add(ll, priorLogProbReference))
            .orElse(priorLogProbReference);

        List<VariableReference> latentVariablesReferences = network.getLatentVertices().stream()
            .map(Variable::getReference)
            .collect(Collectors.toList());

        TensorflowComputableGraph computableGraph = builder.build();

        Optional<Map<VariableReference, VariableReference>> logLikelihoodGradients = logLikelihoodReference.map(
            logLikelihoodOp -> computableGraph.addGradients(
                logLikelihoodOp,
                latentVariablesReferences
            )
        );

        Map<VariableReference, VariableReference> logProbGradients = computableGraph.addGradients(
            logProbReference,
            latentVariablesReferences
        );

        List<Variable<?>> latentVariables = builder.getLatentVariables().stream()
            .map(v -> new TensorflowVariable<>(computableGraph, v))
            .collect(Collectors.toList());

        return new TensorflowProbabilisticGraphWithGradient(
            computableGraph,
            latentVariables,
            logProbReference,
            logLikelihoodReference.orElse(null),
            logProbGradients,
            logLikelihoodGradients.orElse(null)
        );

    }

    private final TensorflowComputableGraph computableGraph;
    private final Map<VariableReference, VariableReference> logProbGradients;
    private final Map<VariableReference, VariableReference> logLikelihoodGradients;

    public TensorflowProbabilisticGraphWithGradient(TensorflowComputableGraph computableGraph,
                                                    List<? extends Variable> latentVariables,
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
    public Map<? extends VariableReference, DoubleTensor> logProbGradients() {
        return logProbGradients(null);
    }

    @Override
    public Map<? extends VariableReference, DoubleTensor> logProbGradients(Map<VariableReference, ?> inputs) {
        return calculateGradients(inputs, logProbGradients);
    }

    @Override
    public Map<? extends VariableReference, DoubleTensor> logLikelihoodGradients() {
        return logLikelihoodGradients(null);
    }

    @Override
    public Map<? extends VariableReference, DoubleTensor> logLikelihoodGradients(Map<VariableReference, ?> inputs) {
        return calculateGradients(inputs, logLikelihoodGradients);
    }

    private Map<? extends VariableReference, DoubleTensor> calculateGradients(Map<VariableReference, ?> inputs,
                                                                              Map<VariableReference, VariableReference> gradientLookup) {

        Map<VariableReference, ?> results = computableGraph.compute(inputs, gradientLookup.keySet());

        Map<VariableReference, DoubleTensor> gradientsByInputName = new HashMap<>();
        for (Map.Entry<VariableReference, ?> result : results.entrySet()) {
            gradientsByInputName.put(gradientLookup.get(result.getKey()), (DoubleTensor) result.getValue());
        }

        return gradientsByInputName;
    }
}
