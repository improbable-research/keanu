package io.improbable.keanu.backend.tensorflow;

import static io.improbable.keanu.backend.ProbabilisticGraph.LOG_PROB;
import static io.improbable.keanu.backend.tensorflow.TensorflowGraphConverter.OpBuilder.binaryOp;
import static io.improbable.keanu.backend.tensorflow.TensorflowGraphConverter.OpBuilder.unaryOp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Shape;
import org.tensorflow.op.Operands;
import org.tensorflow.op.Scope;

import io.improbable.keanu.backend.ProbabilisticGraph;
import io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.ArcTan2Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DifferenceVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DivisionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleBinaryOpVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MatrixMultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MaxVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MinVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.AbsVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ArcCosVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ArcSinVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ArcTanVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.CeilVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.CosVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ExpVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.FloorVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.LogGammaVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.LogVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.MatrixDeterminantVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.MatrixInverseVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.RoundVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SinVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SumVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.TanVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class TensorflowGraphConverter {

    private static Map<Class<?>, OpMapper> opMappers;

    static {
        opMappers = new HashMap<>();

        //binary ops
        opMappers.put(AdditionVertex.class, binaryOp(OpType.ADD));
        opMappers.put(DifferenceVertex.class, binaryOp(OpType.SUBTRACT));
        opMappers.put(DivisionVertex.class, binaryOp(OpType.DIVIDE));
        opMappers.put(MultiplicationVertex.class, binaryOp(OpType.MULTIPLY));
        opMappers.put(MatrixMultiplicationVertex.class, binaryOp(OpType.MATRIX_MULTIPLY));
        opMappers.put(MaxVertex.class, binaryOp(OpType.MAX));
        opMappers.put(MinVertex.class, binaryOp(OpType.MIN));
        opMappers.put(ArcTan2Vertex.class, binaryOp(OpType.ATAN2));
        opMappers.put(PowerVertex.class, binaryOp(OpType.POW));

        //unary ops
        opMappers.put(AbsVertex.class, unaryOp(OpType.ABS));
        opMappers.put(ArcCosVertex.class, unaryOp(OpType.ACOS));
        opMappers.put(ArcSinVertex.class, unaryOp(OpType.ASIN));
        opMappers.put(ArcTanVertex.class, unaryOp(OpType.ATAN));
        opMappers.put(CeilVertex.class, unaryOp(OpType.CEIL));
        opMappers.put(CosVertex.class, unaryOp(OpType.COS));
        opMappers.put(ExpVertex.class, unaryOp(OpType.EXP));
        opMappers.put(FloorVertex.class, unaryOp(OpType.FLOOR));
        opMappers.put(LogGammaVertex.class, unaryOp(OpType.LOG_GAMMA));
        opMappers.put(LogVertex.class, unaryOp(OpType.LOG));
        opMappers.put(MatrixDeterminantVertex.class, unaryOp(OpType.MATRIX_DETERMINANT));
        opMappers.put(MatrixInverseVertex.class, unaryOp(OpType.MATRIX_INVERSE));
        opMappers.put(RoundVertex.class, unaryOp(OpType.ROUND));
        opMappers.put(Sigmoid.class, unaryOp(OpType.SIGMOID));
        opMappers.put(SinVertex.class, unaryOp(OpType.SIN));
        opMappers.put(TanVertex.class, unaryOp(OpType.TAN));

        //special case ops
        opMappers.put(ConstantDoubleVertex.class, TensorflowGraphConverter::createDoubleConstant);
        opMappers.put(SumVertex.class, TensorflowGraphConverter::createSum);
        opMappers.put(ConcatenationVertex.class, TensorflowGraphConverter::createConcat);
    }

    interface OpMapper {
        Output<?> apply(Vertex<?> vertex, Map<Vertex<?>, Output<?>> lookup, GraphBuilder graphBuilder);
    }

    static class OpBuilder {
        static OpMapper binaryOp(OpType op) {
            return (vertex, lookup, graphBuilder) -> {
                DoubleBinaryOpVertex binaryOpVertex = (DoubleBinaryOpVertex) vertex;
                Output<?> leftOperand = lookup.get(binaryOpVertex.getLeft());
                Output<?> rightOperand = lookup.get(binaryOpVertex.getRight());
                return graphBuilder.binaryOp(op, getTensorflowOpName(binaryOpVertex), leftOperand, rightOperand);
            };
        }

        static OpMapper unaryOp(OpType op) {
            return (vertex, lookup, graphBuilder) -> {
                DoubleUnaryOpVertex unaryOpVertex = (DoubleUnaryOpVertex) vertex;
                Output<?> operand = lookup.get(unaryOpVertex.getInputVertex());
                return graphBuilder.unaryOp(op, getTensorflowOpName(unaryOpVertex), operand);
            };
        }
    }

    private static Output<?> createConcat(Vertex<?> vertex, Map<Vertex<?>, Output<?>> lookup, GraphBuilder graphBuilder) {
        ConcatenationVertex concatenationVertex = (ConcatenationVertex) vertex;

        Output<Double>[] inputs = (Output<Double>[]) Operands.asOutputs(
            Arrays.stream(concatenationVertex.getOperands())
                .map(v -> lookup.get(v))
                .collect(Collectors.toList())
        );

        return graphBuilder.concat(inputs, concatenationVertex.getDimension(), getTensorflowOpName(concatenationVertex));
    }

    private static <T> Output<T> createSum(Vertex<?> vertex, Map<Vertex<?>, Output<?>> lookup, GraphBuilder graphBuilder) {
        SumVertex summationVertex = (SumVertex) vertex;
        Output<?> input = lookup.get(summationVertex.getInputVertex());
        String name = getTensorflowOpName(vertex);

        int dims = input.shape().numDimensions();
        Output<Integer> dimRange = graphBuilder.constant(TensorShape.dimensionRange(0, dims), new long[]{dims}, name + "_dimRange");

        return graphBuilder.binaryOp(OpType.SUM, name, input, dimRange);
    }

    private static Output<Double> createDoubleConstant(Vertex<?> vertex, Map<Vertex<?>, Output<?>> lookup, GraphBuilder graphBuilder) {

        Object value = vertex.getValue();

        if (value instanceof DoubleTensor) {
            DoubleTensor doubleValue = (DoubleTensor) value;
            return graphBuilder.constant(doubleValue.asFlatDoubleArray(), doubleValue.getShape(), getTensorflowOpName(vertex));
        } else {
            throw new IllegalArgumentException("Can only convert doubles at the moment");
        }

    }

    private static Output<?> createPlaceholder(Vertex<?> vertex, GraphBuilder graphBuilder) {

        Object value = vertex.getValue();

        if (value instanceof DoubleTensor) {

            long[] shape = ((DoubleTensor) value).getShape();
            String tensorflowOpName = getTensorflowOpName(vertex);

            return graphBuilder.placeholder(tensorflowOpName, toShape(shape), Double.class);
        }

        throw new IllegalArgumentException("Can only convert doubles at the moment");
    }

    private static Shape toShape(long[] shape) {
        return Shape.make(shape[0], Arrays.copyOfRange(shape, 1, shape.length));
    }

    private static String getTensorflowOpName(Vertex vertex) {

        String name = vertex.getLabel() != null ? vertex.getLabel().toString() : vertex.getId().toString();
        return name.replace("]", "").replace("[", "").replace(",", "_");
    }

    public static ProbabilisticGraph convert(BayesianNetwork network) {

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

        addLogProbSum(logProbOps, graphBuilder);

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

    private static void addLogProbSum(List<Output<Double>> logProbOps, GraphBuilder graphBuilder) {
        Output<Double> totalLogProb = logProbOps.get(0);
        for (int i = 1; i < logProbOps.size(); i++) {

            Output<Double> logProbContrib = logProbOps.get(i);
            String name = i == logProbOps.size() - 1 ? LOG_PROB : "logProb" + i;
            totalLogProb = graphBuilder.add(totalLogProb, logProbContrib, name);
        }
    }

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
