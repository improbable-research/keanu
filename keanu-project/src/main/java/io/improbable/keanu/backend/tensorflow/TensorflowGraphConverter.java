package io.improbable.keanu.backend.tensorflow;

import static io.improbable.keanu.backend.ProbabilisticGraph.LOG_PROB;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.stream.Collectors;

import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Shape;
import org.tensorflow.op.Scope;

import io.improbable.keanu.backend.ProbabilisticGraph;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DifferenceVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DivisionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleBinaryOpVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MatrixMultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.LogVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SumVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class TensorflowGraphConverter {

    interface OpMapper {
        Output<?> apply(Vertex<?> vertex, Map<Vertex<?>, Output<?>> lookup, GraphBuilder graphBuilder);
    }

    private static Map<Class<?>, OpMapper> opMappers;

    static {
        opMappers = new HashMap<>();

        opMappers.put(ConstantDoubleVertex.class, TensorflowGraphConverter::createDoubleConstant);
        opMappers.put(AdditionVertex.class, TensorflowGraphConverter::createAddition);
        opMappers.put(DifferenceVertex.class, TensorflowGraphConverter::createSubtraction);
        opMappers.put(DivisionVertex.class, TensorflowGraphConverter::createDivision);
        opMappers.put(MultiplicationVertex.class, TensorflowGraphConverter::createMultiplication);
        opMappers.put(MatrixMultiplicationVertex.class, TensorflowGraphConverter::createMatrixMultiplication);

        opMappers.put(PowerVertex.class, TensorflowGraphConverter::createPow);

        opMappers.put(LogVertex.class, TensorflowGraphConverter::createLog);
        opMappers.put(SumVertex.class, TensorflowGraphConverter::createSum);
    }

    private static Output<?> createSum(Vertex<?> vertex, Map<Vertex<?>, Output<?>> lookup, GraphBuilder graphBuilder) {
        return createUnaryOp(vertex, lookup, graphBuilder::reduceSum);
    }

    private static Output<?> createLog(Vertex<?> vertex, Map<Vertex<?>, Output<?>> lookup, GraphBuilder graphBuilder) {
        return createUnaryOp(vertex, lookup, graphBuilder::log);
    }

    private static Output<?> createAddition(Vertex<?> vertex, Map<Vertex<?>, Output<?>> lookup, GraphBuilder graphBuilder) {
        return createBinaryOp(vertex, lookup, graphBuilder::add);
    }

    private static Output<?> createSubtraction(Vertex<?> vertex, Map<Vertex<?>, Output<?>> lookup, GraphBuilder graphBuilder) {
        return createBinaryOp(vertex, lookup, graphBuilder::sub);
    }

    private static Output<?> createMultiplication(Vertex<?> vertex, Map<Vertex<?>, Output<?>> lookup, GraphBuilder graphBuilder) {
        return createBinaryOp(vertex, lookup, graphBuilder::mul);
    }

    private static Output<?> createDivision(Vertex<?> vertex, Map<Vertex<?>, Output<?>> lookup, GraphBuilder graphBuilder) {
        return createBinaryOp(vertex, lookup, graphBuilder::div);
    }

    private static Output<?> createMatrixMultiplication(Vertex<?> vertex, Map<Vertex<?>, Output<?>> lookup, GraphBuilder graphBuilder) {
        return createBinaryOp(vertex, lookup, graphBuilder::mmul);
    }

    private static Output<?> createPow(Vertex<?> vertex, Map<Vertex<?>, Output<?>> lookup, GraphBuilder graphBuilder) {
        return createBinaryOp(vertex, lookup, graphBuilder::pow);
    }

    private static Output<Double> createDoubleConstant(Vertex<?> vertex, Map<Vertex<?>, Output<?>> lookup, GraphBuilder graphBuilder) {

        Object value = vertex.getValue();

        if (value instanceof DoubleTensor) {
            DoubleTensor doubleValue = (DoubleTensor) value;
            return graphBuilder.constant(doubleValue.asFlatDoubleArray(), toLongs(doubleValue.getShape()), getTensorflowOpName(vertex));
        } else {
            throw new IllegalArgumentException("Can only convert doubles at the moment");
        }

    }

    private static Output<?> createPlaceholder(Vertex<?> vertex, GraphBuilder graphBuilder) {

        Object value = vertex.getValue();

        if (value instanceof DoubleTensor) {

            int[] shape = ((DoubleTensor) value).getShape();
            String tensorflowOpName = getTensorflowOpName(vertex);

            return graphBuilder.placeholder(tensorflowOpName, toShape(shape), Double.class);
        }

        throw new IllegalArgumentException("Can only convert doubles at the moment");
    }

    private static Shape toShape(int[] intShape) {
        long[] restOfShape = new long[intShape.length - 1];
        for (int i = 1; i < intShape.length; i++) {
            restOfShape[i - 1] = intShape[i];
        }
        return Shape.make(intShape[0], restOfShape);
    }

    interface GraphBuilderUnaryOp {
        Output<?> apply(Output<Double> operand, String name);
    }

    private static Output<?> createUnaryOp(Vertex<?> vertex, Map<Vertex<?>, Output<?>> lookup, GraphBuilderUnaryOp op) {
        DoubleUnaryOpVertex unaryOpVertex = (DoubleUnaryOpVertex) vertex;

        Output<Double> operand = (Output<Double>) lookup.get(unaryOpVertex.getInputVertex());

        return op.apply(operand, getTensorflowOpName(unaryOpVertex));
    }

    interface GraphBuilderBinaryOp {
        Output<?> apply(Output<Double> leftOperand, Output<Double> rightOperand, String name);
    }

    private static Output<?> createBinaryOp(Vertex<?> vertex, Map<Vertex<?>, Output<?>> lookup, GraphBuilderBinaryOp op) {
        DoubleBinaryOpVertex binaryOpVertex = (DoubleBinaryOpVertex) vertex;
        Output<Double> leftOperand = (Output<Double>) lookup.get(binaryOpVertex.getLeft());
        Output<Double> rightOperand = (Output<Double>) lookup.get(binaryOpVertex.getRight());
        return op.apply(leftOperand, rightOperand, getTensorflowOpName(binaryOpVertex));
    }

    private static long[] toLongs(int[] ints) {
        long[] longs = new long[ints.length];
        for (int i = 0; i < ints.length; i++) {
            longs[i] = ints[i];
        }
        return longs;
    }

    private static String getTensorflowOpName(Vertex vertex) {

        String name = vertex.getLabel() != null ? vertex.getLabel().toString() : vertex.getId().toString();
        return name.replace("]", "").replace("[", "").replace(",", "_");
    }

    public static ProbabilisticGraph convert(BayesianNetwork network) {

//        addLogProb(network);

        Graph graph = new Graph();
        Scope scope = new Scope(graph);
        GraphBuilder graphBuilder = new GraphBuilder(scope);

        Set<Vertex> queuedAlready = network.getLatentVertices().get(0).getConnectedGraph();
        PriorityQueue<Vertex> priorityQueue = new PriorityQueue<>(Comparator.comparing(Vertex::getId, Comparator.naturalOrder()));
        priorityQueue.addAll(queuedAlready);

        Map<Vertex<?>, Output<?>> lookup = new HashMap<>();
        List<String> placeHolderOps = new ArrayList<>();
        List<Output<Double>> logProbOps = new ArrayList<>();

        Vertex visiting;
        while ((visiting = priorityQueue.poll()) != null) {

            if (visiting instanceof GaussianVertex) {
                if (visiting.isObserved()) {
                    lookup.put(visiting, createDoubleConstant(visiting, lookup, graphBuilder));
                } else {
                    Output<?> tfVisiting = createPlaceholder(visiting, graphBuilder);
                    lookup.put(visiting, tfVisiting);
                    placeHolderOps.add(tfVisiting.op().name());
                }

                Output<Double> logProbFromVisiting = addLogProbFrom((GaussianVertex) visiting, lookup, graphBuilder);
                logProbOps.add(logProbFromVisiting);

            } else {
                OpMapper vertexMapper = opMappers.get(visiting.getClass());

                if (vertexMapper == null) {
                    throw new IllegalArgumentException("Vertex type " + visiting.getClass() + " not supported");
                }

                lookup.put(visiting, vertexMapper.apply(visiting, lookup, graphBuilder));
            }
        }

        Output<Double> totalLogProb = logProbOps.get(0);
        for (int i = 1; i < logProbOps.size(); i++) {

            Output<Double> logProbContrib = logProbOps.get(i);
            String name = i == logProbOps.size() - 1 ? LOG_PROB : "logProb" + i;
            totalLogProb = graphBuilder.add(totalLogProb, logProbContrib, name);
        }

        Map<String, Output<?>> gradientByInput = addGradients(graph, placeHolderOps);

        return new TensorflowProbabilisticGraph(
            new Session(graph),
            placeHolderOps,
            gradientByInput
        );
    }

    private static Output<Double> addLogProbFrom(GaussianVertex vertex, Map<Vertex<?>, Output<?>> lookup, GraphBuilder graphBuilder) {

        LogProbGraph logProbGraph = vertex.logProbGraph();
        Map<Vertex<?>, Vertex<?>> inputs = logProbGraph.getInputs();

        //setup graph connection
        for (Map.Entry<Vertex<?>, Vertex<?>> input : inputs.entrySet()) {
            lookup.put(input.getValue(), lookup.get(input.getKey()));
        }

        List<Vertex<?>> topoSortedVertices = ((Set<Vertex<?>>) logProbGraph.getLogProbOutput().getConnectedGraph()).stream()
            .sorted(Comparator.comparing(Vertex::getId))
            .collect(Collectors.toList());

        HashSet<Vertex<?>> logProbInputs = new HashSet<>(inputs.values());

        for (Vertex<?> visiting : topoSortedVertices) {

            if (!logProbInputs.contains(visiting)) {

                OpMapper vertexMapper = opMappers.get(visiting.getClass());

                if (vertexMapper == null) {
                    throw new IllegalArgumentException("Vertex type " + visiting.getClass() + " not supported");
                }

                lookup.put(visiting, vertexMapper.apply(visiting, lookup, graphBuilder));
            }
        }

        return (Output<Double>) lookup.get(logProbGraph.getLogProbOutput());
    }

//    private static void addLogProb(BayesianNetwork network) {
//
//        List<Vertex> latentOrObservedVertices = network.getLatentOrObservedVertices();
//
//        DoubleVertex logProb = null;
//        for (Vertex v : latentOrObservedVertices) {
//
//            if (v instanceof Probabilistic) {
//                if (v instanceof GaussianVertex) {
//                    GaussianVertex gaussianVertex = ((GaussianVertex) v);
//
//                    DoubleVertex logProbGraph = gaussianVertex.logProbGraph(gaussianVertex);
//
//                    if (logProb == null) {
//                        logProb = logProbGraph;
//                    } else {
//                        logProb = logProb.plus(logProbGraph);
//                    }
//                } else {
//                    throw new IllegalArgumentException("Only supports log prob of Gaussian distributions");
//                }
//            }
//        }
//
//        if (logProb != null) {
//            logProb.setLabel(new VertexLabel(LOG_PROB));
//        }
//    }

    private static Map<String, Output<?>> addGradients(Graph graph, List<String> placeHolderOps) {

        Output<?>[] wrt = placeHolderOps.stream()
            .map(opName -> graph.operation(opName).output(0))
            .toArray(Output[]::new);

        Output<?>[] gradientOutputs = graph.addGradients(graph.operation(LOG_PROB).output(0), wrt);

        Map<String, Output<?>> gradientByInput = new HashMap<>();
        for (int i = 0; i < placeHolderOps.size(); i++) {
            gradientByInput.put(placeHolderOps.get(i), gradientOutputs[i]);
        }

        return gradientByInput;
    }

}
