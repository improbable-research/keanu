package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.backend.ProbabilisticGraphConverter;
import io.improbable.keanu.network.BayesianNetwork;


public class TensorflowProbabilisticGraphFactory {

    public static TensorflowProbabilisticGraph convert(BayesianNetwork network) {
        return ProbabilisticGraphConverter.convert(network, new TensorflowProbabilisticGraphBuilder());
    }

//    public static TensorflowProbabilisticGraph convert2(BayesianNetwork network) {
//
//        TensorflowGraphBuilder graphBuilder = new TensorflowGraphBuilder();
//        TensorflowComputableGraph computableGraph = TensorflowComputableGraphFactory.create(network.getVertices().get(0).getConnectedGraph(), graphBuilder);
//
//        Map<Vertex<?>, Output<?>> vertexLookup = graphBuilder.getLookup();
//        TensorflowOpHelper opHelper = new TensorflowOpHelper(computableGraph.getScope());
//
//        Output<Double> priorLogProbOutput = addLogProbCalculation(
//            opHelper,
//            vertexLookup,
//            network.getLatentVertices()
//        );
//
//        Output<Double> totalLogProbOutput;
//        StringVariableReference logLikelihoodReference = null;
//
//        if (!network.getObservedVertices().isEmpty()) {
//            Output<Double> logLikelihoodOutput = addLogProbCalculation(
//                opHelper,
//                vertexLookup,
//                network.getObservedVertices()
//            );
//
//            logLikelihoodReference = new StringVariableReference(logLikelihoodOutput.op().name());
//            totalLogProbOutput = opHelper.add(logLikelihoodOutput, priorLogProbOutput);
//        } else {
//            totalLogProbOutput = priorLogProbOutput;
//        }
//
//        List<Variable<?>> latentVariables = network.getLatentVertices().stream()
//            .map(v -> new TensorflowVariable<>(computableGraph, v.getReference()))
//            .collect(Collectors.toList());
//
//        StringVariableReference logProbReference = new StringVariableReference(totalLogProbOutput.op().name());
//
//        return new TensorflowProbabilisticGraph(computableGraph, latentVariables, logProbReference, logLikelihoodReference);
//    }
//
//    private static Output<Double> addLogProbCalculation(TensorflowOpHelper graphBuilder,
//                                                        Map<Vertex<?>, Output<?>> vertexLookup,
//                                                        List<Vertex> probabilisticVertices) {
//        List<Output<Double>> logProbOps = new ArrayList<>();
//
//        for (Vertex visiting : probabilisticVertices) {
//
//            if (visiting instanceof LogProbAsAGraphable) {
//                LogProbGraph logProbGraph = ((LogProbAsAGraphable) visiting).logProbGraph();
//                Output<Double> logProbFromVisiting = addLogProbFrom(logProbGraph, vertexLookup, graphBuilder);
//                logProbOps.add(logProbFromVisiting);
//
//            } else {
//                throw new IllegalArgumentException("Vertex type " + visiting.getClass() + " logProb as a graph not supported");
//            }
//        }
//
//        return addLogProbSumTotal(logProbOps, graphBuilder);
//    }
//
//    private static Output<Double> addLogProbFrom(LogProbGraph logProbGraph, Map<Vertex<?>, Output<?>> lookup, TensorflowOpHelper graphBuilder) {
//
//        Map<Vertex<?>, Vertex<?>> inputs = logProbGraph.getInputs();
//
//        //setup graph connection
//        for (Map.Entry<Vertex<?>, Vertex<?>> input : inputs.entrySet()) {
//            lookup.put(input.getValue(), lookup.get(input.getKey()));
//        }
//
//        List<Vertex> topoSortedVertices = (logProbGraph.getLogProbOutput().getConnectedGraph()).stream()
//            .sorted(Comparator.comparing(Vertex::getId))
//            .collect(Collectors.toList());
//
//        HashSet<Vertex<?>> logProbInputs = new HashSet<>(inputs.values());
//
//        for (Vertex visiting : topoSortedVertices) {
//
//            if (!logProbInputs.contains(visiting)) {
//
//                TensorflowGraphConverter.OpMapper vertexMapper = TensorflowGraphConverter.opMappers.get(visiting.getClass());
//
//                if (vertexMapper == null) {
//                    throw new IllegalArgumentException("Vertex type " + visiting.getClass() + " not supported");
//                }
//
//                lookup.put(visiting, vertexMapper.apply(visiting, lookup, graphBuilder));
//            }
//        }
//
//        return (Output<Double>) lookup.get(logProbGraph.getLogProbOutput());
//    }
//
//    private static Output<Double> addLogProbSumTotal(List<Output<Double>> logProbOps, TensorflowOpHelper graphBuilder) {
//
//        Output<Double> totalLogProb = logProbOps.get(0);
//
//        if (logProbOps.size() == 1) {
//            return totalLogProb;
//        }
//
//        for (int i = 1; i < logProbOps.size() - 1; i++) {
//            Output<Double> logProbContrib = logProbOps.get(i);
//            totalLogProb = graphBuilder.add(totalLogProb, logProbContrib);
//        }
//
//        Output<Double> lastLogProbContrib = logProbOps.get(logProbOps.size() - 1);
//        return graphBuilder.add(totalLogProb, lastLogProbContrib);
//    }

}
