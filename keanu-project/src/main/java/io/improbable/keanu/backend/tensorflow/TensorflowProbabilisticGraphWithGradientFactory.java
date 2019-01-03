package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.backend.Variable;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import org.tensorflow.Graph;
import org.tensorflow.Output;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class TensorflowProbabilisticGraphWithGradientFactory {

    public static TensorflowProbabilisticGraphWithGradient convert(BayesianNetwork bayesianNetwork) {

        Map<Vertex<?>, Output<?>> vertexLookup = new HashMap<>();
        TensorflowProbabilisticGraph tensorflowProbabilisticGraph = TensorflowProbabilisticGraphFactory.convert(bayesianNetwork, vertexLookup);

        TensorflowComputableGraph computeGraph = tensorflowProbabilisticGraph.getComputableGraph();

        List<Vertex<DoubleTensor>> latentVertices = bayesianNetwork.getContinuousLatentVertices();

        List<VariableReference> latentVariablesReferences = latentVertices.stream()
            .map(Vertex::getReference)
            .collect(Collectors.toList());

        List<Variable<?>> latentVariables = latentVariablesReferences.stream()
            .map(ref -> new TensorflowVariable<>(computeGraph, ref))
            .collect(Collectors.toList());

        Graph computableGraph = tensorflowProbabilisticGraph.getComputableGraph().getScope().graph();

        Map<VariableReference, VariableReference> logLikelihoodGradients = null;
        VariableReference logLikelihoodOp = tensorflowProbabilisticGraph.getLogLikelihoodOp();

        if (logLikelihoodOp != null) {
            logLikelihoodGradients = addGradients(
                computableGraph,
                tensorflowProbabilisticGraph.getLogLikelihoodOp(),
                latentVariablesReferences
            );
        }

        Map<VariableReference, VariableReference> logProbGradients = addGradients(
            computableGraph,
            tensorflowProbabilisticGraph.getLogProbOp(),
            latentVariablesReferences
        );

        return new TensorflowProbabilisticGraphWithGradient(
            tensorflowProbabilisticGraph.getComputableGraph(),
            latentVariables,
            tensorflowProbabilisticGraph.getLogProbOp(),
            tensorflowProbabilisticGraph.getLogLikelihoodOp(),
            logProbGradients,
            logLikelihoodGradients
        );
    }

    /**
     * @param graph               graph to add gradients
     * @param ofLabel             of operation reference
     * @param withRespectToLabels with respect to references
     * @return gradient output name to input name lookup
     */
    private static Map<VariableReference, VariableReference> addGradients(Graph graph,
                                                                          VariableReference ofLabel,
                                                                          List<VariableReference> withRespectToLabels) {

        Output<?>[] wrt = withRespectToLabels.stream()
            .map(opName -> graph.operation(opName.toStringReference()).output(0))
            .toArray(Output[]::new);

        Output<?>[] gradientOutputs = graph.addGradients(graph.operation(ofLabel.toStringReference()).output(0), wrt);

        Map<VariableReference, VariableReference> gradientOutputNameToInputName = new HashMap<>();
        for (int i = 0; i < withRespectToLabels.size(); i++) {
            gradientOutputNameToInputName.put(new StringVariableReference(gradientOutputs[i].op().name()), withRespectToLabels.get(i));
        }

        return gradientOutputNameToInputName;
    }
}
