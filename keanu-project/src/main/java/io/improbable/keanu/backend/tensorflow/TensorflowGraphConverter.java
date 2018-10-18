package io.improbable.keanu.backend.tensorflow;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.bytedeco.javacpp.Loader;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Shape;
import org.tensorflow.op.Operands;
import org.tensorflow.op.Scope;

import io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LogProbAsAGraphable;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexBinaryOp;
import io.improbable.keanu.vertices.VertexUnaryOp;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.AndBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.OrBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.NotVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleIfVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.ArcTan2Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DifferenceVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DivisionVertex;
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
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerAdditionVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerDifferenceVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerDivisionVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMaxVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMinVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMultiplicationVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerPowerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerAbsVertex;

public class TensorflowGraphConverter {

    private static Map<Class<?>, OpMapper> opMappers;

    static {
        Loader.load(org.bytedeco.javacpp.tensorflow.class);
        opMappers = new HashMap<>();

        //double binary ops
        opMappers.put(AdditionVertex.class, binaryOp(OpType.ADD));
        opMappers.put(DifferenceVertex.class, binaryOp(OpType.SUBTRACT));
        opMappers.put(DivisionVertex.class, binaryOp(OpType.DIVIDE));
        opMappers.put(MultiplicationVertex.class, binaryOp(OpType.MULTIPLY));
        opMappers.put(MatrixMultiplicationVertex.class, binaryOp(OpType.MATRIX_MULTIPLY));
        opMappers.put(MaxVertex.class, binaryOp(OpType.MAX));
        opMappers.put(MinVertex.class, binaryOp(OpType.MIN));
        opMappers.put(ArcTan2Vertex.class, binaryOp(OpType.ATAN2));
        opMappers.put(PowerVertex.class, binaryOp(OpType.POW));

        //double unary ops
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

        //bool binary ops
        opMappers.put(AndBinaryVertex.class, binaryOp(OpType.AND));
        opMappers.put(OrBinaryVertex.class, binaryOp(OpType.OR));
        opMappers.put(NotVertex.class, unaryOp(OpType.NOT));

        //integer binary ops
        opMappers.put(IntegerAdditionVertex.class, binaryOp(OpType.ADD));
        opMappers.put(IntegerDifferenceVertex.class, binaryOp(OpType.SUBTRACT));
        opMappers.put(IntegerDivisionVertex.class, binaryOp(OpType.DIVIDE));
        opMappers.put(IntegerMultiplicationVertex.class, binaryOp(OpType.MULTIPLY));
        opMappers.put(IntegerPowerVertex.class, binaryOp(OpType.POW));
        opMappers.put(IntegerMaxVertex.class, binaryOp(OpType.MAX));
        opMappers.put(IntegerMinVertex.class, binaryOp(OpType.MIN));

        //integer unary ops
        opMappers.put(IntegerAbsVertex.class, unaryOp(OpType.ABS));

        //constants
        opMappers.put(ConstantDoubleVertex.class, TensorflowGraphConverter::createConstant);
        opMappers.put(ConstantIntegerVertex.class, TensorflowGraphConverter::createConstant);
        opMappers.put(ConstantBoolVertex.class, TensorflowGraphConverter::createConstant);

        //special case ops
        opMappers.put(DoubleIfVertex.class, TensorflowGraphConverter::createDoubleIf);
        opMappers.put(SumVertex.class, TensorflowGraphConverter::createSum);
        opMappers.put(ConcatenationVertex.class, TensorflowGraphConverter::createConcat);
    }

    interface OpMapper {
        Output<?> apply(Vertex<?> vertex, Map<Vertex<?>, Output<?>> lookup, GraphBuilder graphBuilder);
    }

    private static OpMapper binaryOp(OpType op) {
        return (vertex, lookup, graphBuilder) -> {
            VertexBinaryOp<?, ?> binaryOpVertex = (VertexBinaryOp<?, ?>) vertex;
            Output<?> leftOperand = lookup.get(binaryOpVertex.getLeft());
            Output<?> rightOperand = lookup.get(binaryOpVertex.getRight());
            return graphBuilder.binaryOp(op, getTensorflowOpName(vertex), leftOperand, rightOperand);
        };
    }

    private static OpMapper unaryOp(OpType op) {
        return (vertex, lookup, graphBuilder) -> {
            VertexUnaryOp unaryOpVertex = (VertexUnaryOp) vertex;
            Output<?> operand = lookup.get(unaryOpVertex.getInput());
            return graphBuilder.unaryOp(op, getTensorflowOpName(vertex), operand);
        };
    }

    private static Output<?> createDoubleIf(Vertex<?> vertex, Map<Vertex<?>, Output<?>> lookup, GraphBuilder graphBuilder) {
        DoubleIfVertex doubleIfVertex = (DoubleIfVertex) vertex;

        Output<Boolean> predicate = (Output<Boolean>) lookup.get(doubleIfVertex.getPredicate());
        Output<Double> thn = (Output<Double>) lookup.get(doubleIfVertex.getThn());
        Output<Double> els = (Output<Double>) lookup.get(doubleIfVertex.getEls());

        long[] predicateShape = doubleIfVertex.getPredicate().getShape();
        Output<Long> shape = graphBuilder.constant(predicateShape, new long[]{predicateShape.length});

        Output<Double> thnBroadcast = graphBuilder.broadcastTo(thn, shape);
        Output<Double> elsBroadcast = graphBuilder.broadcastTo(els, shape);

        return graphBuilder.where(predicate, thnBroadcast, elsBroadcast);
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
        Output<?> input = lookup.get(summationVertex.getInput());
        String name = getTensorflowOpName(vertex);

        int dims = input.shape().numDimensions();
        Output<Integer> dimRange = graphBuilder.constant(TensorShape.dimensionRange(0, dims), new long[]{dims});

        return graphBuilder.binaryOp(OpType.SUM, name, input, dimRange);
    }

    private static Output<?> createConstant(Vertex<?> vertex, Map<Vertex<?>, Output<?>> lookup, GraphBuilder graphBuilder) {

        Object value = vertex.getValue();

        if (value instanceof DoubleTensor) {
            DoubleTensor doubleValue = (DoubleTensor) value;
            return graphBuilder.constant(doubleValue.asFlatDoubleArray(), doubleValue.getShape(), getTensorflowOpName(vertex));
        } else if (value instanceof IntegerTensor) {
            IntegerTensor integerValue = (IntegerTensor) value;
            return graphBuilder.constant(integerValue.asFlatIntegerArray(), integerValue.getShape(), getTensorflowOpName(vertex));
        } else if (value instanceof BooleanTensor) {
            BooleanTensor booleanValue = (BooleanTensor) value;
            return graphBuilder.constant(booleanValue.asFlatArray(), booleanValue.getShape(), getTensorflowOpName(vertex));
        }

        throw new IllegalArgumentException("Cannot convert " + value.getClass());
    }

    private static Output<?> createPlaceholder(Vertex<?> vertex, GraphBuilder graphBuilder) {

        Object value = vertex.getValue();

        if (value instanceof DoubleTensor) {
            return graphBuilder.placeholder(getTensorflowOpName(vertex), toShape(vertex.getShape()), Double.class);
        } else if (value instanceof IntegerTensor) {
            return graphBuilder.placeholder(getTensorflowOpName(vertex), toShape(vertex.getShape()), Integer.class);
        } else if (value instanceof BooleanTensor) {
            return graphBuilder.placeholder(getTensorflowOpName(vertex), toShape(vertex.getShape()), Boolean.class);
        }

        throw new IllegalArgumentException("Cannot convert " + value.getClass());
    }

    private static Shape toShape(long[] shape) {
        return Shape.make(shape[0], Arrays.copyOfRange(shape, 1, shape.length));
    }

    private static String getTensorflowOpName(Vertex vertex) {

        String name = vertex.getLabel() != null ? vertex.getLabel().toString() : vertex.getId().toString();
        return name.replace("]", "").replace("[", "").replace(",", "_");
    }

    public static TensorflowComputableGraph convert(Set<Vertex> queuedAlready) {
        return convert(queuedAlready, new HashMap<>());
    }

    public static TensorflowComputableGraph convert(Collection<? extends Vertex> queuedAlready, Map<Vertex<?>, Output<?>> lookup) {

        Graph graph = new Graph();
        Scope scope = new Scope(graph);
        TensorflowComputableGraph computableGraph = new TensorflowComputableGraph(new Session(scope.graph()), scope);
        GraphBuilder graphBuilder = computableGraph.getGraphBuilder();

        PriorityQueue<Vertex> priorityQueue = new PriorityQueue<>(Comparator.comparing(Vertex::getId, Comparator.naturalOrder()));
        priorityQueue.addAll(queuedAlready);

        Vertex visiting;
        while ((visiting = priorityQueue.poll()) != null) {

            if (visiting instanceof Probabilistic) {
                if (visiting.isObserved()) {
                    lookup.put(visiting, createConstant(visiting, lookup, graphBuilder));
                } else {
                    Output<?> tfVisiting = createPlaceholder(visiting, graphBuilder);
                    lookup.put(visiting, tfVisiting);
                }
            } else {
                OpMapper vertexMapper = opMappers.get(visiting.getClass());

                if (vertexMapper == null) {
                    throw new IllegalArgumentException("Vertex type " + visiting.getClass() + " not supported");
                }

                lookup.put(visiting, vertexMapper.apply(visiting, lookup, graphBuilder));
            }
        }

        return computableGraph;
    }

    public static TensorflowProbabilisticGraph convert(BayesianNetwork network) {
        return convert(network, new HashMap<>());
    }

    public static TensorflowProbabilisticGraph convert(BayesianNetwork network, Map<Vertex<?>, Output<?>> vertexLookup) {

        TensorflowComputableGraph computableGraph = convert(network.getVertices(), vertexLookup);

        return addLogProbCalculation(
            computableGraph,
            vertexLookup,
            network.getLatentOrObservedVertices()
        );
    }

    public static TensorflowProbabilisticWithGradientGraph convertWithGradient(BayesianNetwork bayesianNetwork) {

        Map<Vertex<?>, Output<?>> vertexLookup = new HashMap<>();
        TensorflowProbabilisticGraph tensorflowProbabilisticGraph = convert(bayesianNetwork, vertexLookup);

        List<String> latentVariables = bayesianNetwork.getContinuousLatentVertices().stream()
            .map(latent -> vertexLookup.get(latent).op().name())
            .collect(Collectors.toList());

        Map<String, String> gradientOutputNameToInputName = addGradients(
            tensorflowProbabilisticGraph.getComputableGraph().getScope().graph(),
            tensorflowProbabilisticGraph.getLogProbSumTotalOpName(),
            latentVariables
        );

        return new TensorflowProbabilisticWithGradientGraph(
            tensorflowProbabilisticGraph.getComputableGraph(),
            tensorflowProbabilisticGraph.getLogProbSumTotalOpName(),
            gradientOutputNameToInputName
        );
    }

    private static TensorflowProbabilisticGraph addLogProbCalculation(TensorflowComputableGraph computableGraph,
                                                                      Map<Vertex<?>, Output<?>> vertexLookup,
                                                                      List<Vertex> latentOrObservedVertices) {
        List<String> latentVariables = new ArrayList<>();
        List<Output<Double>> logProbOps = new ArrayList<>();

        for (Vertex visiting : latentOrObservedVertices) {

            if (visiting instanceof LogProbAsAGraphable) {
                if (!visiting.isObserved()) {
                    latentVariables.add(vertexLookup.get(visiting).op().name());
                }

                LogProbGraph logProbGraph = ((LogProbAsAGraphable) visiting).logProbGraph();
                Output<Double> logProbFromVisiting = addLogProbFrom(logProbGraph, vertexLookup, computableGraph.getGraphBuilder());
                logProbOps.add(logProbFromVisiting);

            } else {
                throw new IllegalArgumentException("Vertex type " + visiting.getClass() + " logProb as a graph not supported");
            }
        }

        Output<Double> logProbSumTotal = addLogProbSumTotal(logProbOps, computableGraph.getGraphBuilder());

        return new TensorflowProbabilisticGraph(computableGraph, logProbSumTotal.op().name());
    }

    private static Output<Double> addLogProbFrom(LogProbGraph logProbGraph, Map<Vertex<?>, Output<?>> lookup, GraphBuilder graphBuilder) {

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

    private static Output<Double> addLogProbSumTotal(List<Output<Double>> logProbOps, GraphBuilder graphBuilder) {

        Output<Double> totalLogProb = logProbOps.get(0);
        for (int i = 1; i < logProbOps.size() - 1; i++) {
            Output<Double> logProbContrib = logProbOps.get(i);
            totalLogProb = graphBuilder.add(totalLogProb, logProbContrib);
        }

        Output<Double> lastLogProbContrib = logProbOps.get(logProbOps.size() - 1);
        return graphBuilder.add(totalLogProb, lastLogProbContrib);
    }

    /**
     * @param graph               graph to add gradients
     * @param ofLabel             of operation label
     * @param withRespectToLabels with respect to labels
     * @return gradient output name to input name lookup
     */
    private static Map<String, String> addGradients(Graph graph, String ofLabel, List<String> withRespectToLabels) {

        Output<?>[] wrt = withRespectToLabels.stream()
            .map(opName -> graph.operation(opName).output(0))
            .toArray(Output[]::new);

        Output<?>[] gradientOutputs = graph.addGradients(graph.operation(ofLabel).output(0), wrt);

        Map<String, String> gradientOutputNameToInputName = new HashMap<>();
        for (int i = 0; i < withRespectToLabels.size(); i++) {
            gradientOutputNameToInputName.put(gradientOutputs[i].op().name(), withRespectToLabels.get(i));
        }

        return gradientOutputNameToInputName;
    }

}
