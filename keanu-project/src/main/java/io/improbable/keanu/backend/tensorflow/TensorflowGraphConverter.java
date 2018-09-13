package io.improbable.keanu.backend.tensorflow;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;

import org.tensorflow.Graph;
import org.tensorflow.Output;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DifferenceVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DivisionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleBinaryOpVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MatrixMultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class TensorflowGraphConverter {

    private static Map<Class<? extends Vertex<?>>, BiConsumer<Vertex<?>, GraphBuilder>> opMappers;

    static {
        opMappers = new HashMap<>();
        opMappers.put(GaussianVertex.class, TensorflowGraphConverter::createDoubleConstant);
        opMappers.put(ConstantDoubleVertex.class, TensorflowGraphConverter::createDoubleConstant);
        opMappers.put(AdditionVertex.class, TensorflowGraphConverter::createAddition);
        opMappers.put(DifferenceVertex.class, TensorflowGraphConverter::createSubtraction);
        opMappers.put(DivisionVertex.class, TensorflowGraphConverter::createDivision);
        opMappers.put(MultiplicationVertex.class, TensorflowGraphConverter::createMultiplication);
        opMappers.put(MatrixMultiplicationVertex.class, TensorflowGraphConverter::createMatrixMultiplication);
    }

    interface GraphBuilderBinaryOp {
        void op(Output<Double> leftOperand, Output<Double> rightOperand, String name);
    }

    private static void createBinaryOp(Vertex<?> vertex, GraphBuilder graphBuilder, GraphBuilderBinaryOp op) {
        DoubleBinaryOpVertex binaryOpVertex = (DoubleBinaryOpVertex) vertex;
        Output<Double> leftOperand = graphBuilder.getOutput(getName(binaryOpVertex.getLeft()));
        Output<Double> rightOperand = graphBuilder.getOutput(getName(binaryOpVertex.getRight()));
        op.op(leftOperand, rightOperand, getName(binaryOpVertex));
    }

    private static void createAddition(Vertex<?> vertex, GraphBuilder graphBuilder) {
        createBinaryOp(vertex, graphBuilder, graphBuilder::add);
    }

    private static void createSubtraction(Vertex<?> vertex, GraphBuilder graphBuilder) {
        createBinaryOp(vertex, graphBuilder, graphBuilder::sub);
    }

    private static void createMultiplication(Vertex<?> vertex, GraphBuilder graphBuilder) {
        createBinaryOp(vertex, graphBuilder, graphBuilder::mul);
    }

    private static void createDivision(Vertex<?> vertex, GraphBuilder graphBuilder) {
        createBinaryOp(vertex, graphBuilder, graphBuilder::div);
    }

    private static void createMatrixMultiplication(Vertex<?> vertex, GraphBuilder graphBuilder) {
        createBinaryOp(vertex, graphBuilder, graphBuilder::mmul);
    }

    private static void createDoubleConstant(Vertex<?> vertex, GraphBuilder graphBuilder) {

        Object value = vertex.getValue();

        if (value instanceof DoubleTensor) {
            DoubleTensor doubleValue = (DoubleTensor) value;
            graphBuilder.constant(doubleValue.asFlatDoubleArray(), toLongs(doubleValue.getShape()), getName(vertex));
        } else {
            throw new IllegalArgumentException("Can only convert doubles at the moment");
        }

    }

    private static long[] toLongs(int[] ints) {
        long[] longs = new long[ints.length];
        for (int i = 0; i < ints.length; i++) {
            longs[i] = ints[i];
        }
        return longs;
    }

    private static String getName(Vertex vertex) {

        String name = vertex.getLabel() != null ? vertex.getLabel().toString() : vertex.getId().toString();
        String cleanName = name.replace("]", "").replace("[", "").replace(",", "_");
        return cleanName;
    }

    public static Graph convert(BayesianNetwork network) {

        Graph graph = new Graph();
        GraphBuilder graphBuilder = new GraphBuilder(graph);

        Set<Vertex> connectedGraph = network.getLatentVertices().get(0).getConnectedGraph();
        List<Vertex> topoSortedGraph = connectedGraph.stream().sorted(Comparator.comparing(Vertex::getId)).collect(Collectors.toList());

        for (Vertex visiting : topoSortedGraph) {
            BiConsumer<Vertex<?>, GraphBuilder> vertexMapper = opMappers.get(visiting.getClass());

            if (vertexMapper == null) {
                throw new IllegalArgumentException("Vertex type " + visiting.getClass() + " not supported");
            }

            vertexMapper.accept(visiting, graphBuilder);
        }

        return graph;
    }


}
