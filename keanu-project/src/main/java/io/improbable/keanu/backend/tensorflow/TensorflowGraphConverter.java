package io.improbable.keanu.backend.tensorflow;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.function.BiConsumer;

import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Shape;
import org.tensorflow.op.Scope;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
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

    public static final String LOG_PROB_LABEL = "..LOG_PROB..";

    private static Map<Class<?>, BiConsumer<Vertex<?>, GraphBuilder>> opMappers;

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

    private static void createSum(Vertex<?> vertex, GraphBuilder graphBuilder) {
        createUnaryOp(vertex, graphBuilder, graphBuilder::reduceSum);
    }

    private static void createLog(Vertex<?> vertex, GraphBuilder graphBuilder) {
        createUnaryOp(vertex, graphBuilder, graphBuilder::log);
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

    private static void createPow(Vertex<?> vertex, GraphBuilder graphBuilder) {
        createBinaryOp(vertex, graphBuilder, graphBuilder::pow);
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

    private static void createPlaceholder(Vertex<?> vertex, GraphBuilder graphBuilder) {

        Object value = vertex.getValue();

        if (value instanceof DoubleTensor) {

            //TODO: Nasty conversion here
            int[] shape = ((DoubleTensor) value).getShape();
            long[] restOfShape = new long[shape.length - 1];
            for (int i = 1; i < shape.length; i++) {
                restOfShape[i - 1] = shape[i];
            }

            graphBuilder.placeholder(getName(vertex), Shape.make(shape[0], restOfShape), Double.class);
        } else {
            throw new IllegalArgumentException("Can only convert doubles at the moment");
        }

    }

    interface GraphBuilderUnaryOp {
        void op(Output<Double> operand, String name);
    }

    private static void createUnaryOp(Vertex<?> vertex, GraphBuilder graphBuilder, GraphBuilderUnaryOp op) {
        DoubleUnaryOpVertex unaryOpVertex = (DoubleUnaryOpVertex) vertex;
        Output<Double> operand = graphBuilder.getOutput(getName(unaryOpVertex.getInputVertex()));
        op.op(operand, getName(unaryOpVertex));
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

        addLogProb(network);

        Graph graph = new Graph();
        Scope scope = new Scope(graph);
        GraphBuilder graphBuilder = new GraphBuilder(scope);

        Set<Vertex> queuedAlready = network.getLatentVertices().get(0).getConnectedGraph();
        PriorityQueue<Vertex> priorityQueue = new PriorityQueue<>(Comparator.comparing(Vertex::getId, Comparator.naturalOrder()));
        priorityQueue.addAll(queuedAlready);

        Vertex visiting;
        while ((visiting = priorityQueue.poll()) != null) {

            if (visiting instanceof Probabilistic) {
                if (visiting.isObserved()) {
                    createDoubleConstant(visiting, graphBuilder);
                } else {
                    createPlaceholder(visiting, graphBuilder);
                }
            } else {
                BiConsumer<Vertex<?>, GraphBuilder> vertexMapper = opMappers.get(visiting.getClass());

                if (vertexMapper == null) {
                    throw new IllegalArgumentException("Vertex type " + visiting.getClass() + " not supported");
                }

                vertexMapper.accept(visiting, graphBuilder);
            }
        }

        return graph;
    }

    private static void addLogProb(BayesianNetwork network) {

        List<Vertex> latentOrObservedVertices = network.getLatentOrObservedVertices();

        DoubleVertex logProb = null;
        for (Vertex v : latentOrObservedVertices) {

            if (v instanceof Probabilistic) {
                if (v instanceof GaussianVertex) {
                    GaussianVertex gaussianVertex = ((GaussianVertex) v);

                    DoubleVertex logProbGraph = gaussianVertex.logProbGraph(gaussianVertex);

                    if (logProb == null) {
                        logProb = logProbGraph;
                    } else {
                        logProb = logProb.plus(logProbGraph);
                    }
                } else {
                    throw new IllegalArgumentException("Only supports log prob of Gaussian distributions");
                }
            }
        }

        if (logProb != null) {
            logProb.setLabel(new VertexLabel(LOG_PROB_LABEL));
        }
    }


}
