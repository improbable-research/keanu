package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.backend.tensorflow.TensorflowOpHelper.OpType;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
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
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.tensorflow.Output;
import org.tensorflow.Shape;
import org.tensorflow.op.Operands;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

public class TensorflowComputableGraphConverter {

    static Map<Class<?>, OpMapper> opMappers;

    static {
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
        opMappers.put(ConstantDoubleVertex.class, (vertex, lookup, graphBuilder) -> createConstant(vertex, graphBuilder));
        opMappers.put(ConstantIntegerVertex.class, (vertex, lookup, graphBuilder) -> createConstant(vertex, graphBuilder));
        opMappers.put(ConstantBoolVertex.class, (vertex, lookup, graphBuilder) -> createConstant(vertex, graphBuilder));

        //special case ops
        opMappers.put(DoubleIfVertex.class, TensorflowComputableGraphConverter::createDoubleIf);
        opMappers.put(SumVertex.class, TensorflowComputableGraphConverter::createSum);
        opMappers.put(ConcatenationVertex.class, TensorflowComputableGraphConverter::createConcat);
    }

    interface OpMapper {
        Output<?> apply(Vertex<?> vertex, Map<VariableReference, Output<?>> lookup, TensorflowOpHelper graphBuilder);
    }

    private static Output<?> getOutput(Map<VariableReference, Output<?>> lookup, Vertex vertex) {
        return lookup.get(vertex.getReference());
    }

    private static OpMapper binaryOp(OpType op) {
        return (vertex, lookup, graphBuilder) -> {
            VertexBinaryOp<?, ?> binaryOpVertex = (VertexBinaryOp<?, ?>) vertex;
            Output<?> leftOperand = getOutput(lookup, binaryOpVertex.getLeft());
            Output<?> rightOperand = getOutput(lookup, binaryOpVertex.getRight());
            return graphBuilder.binaryOp(op, getTensorflowOpName(vertex), leftOperand, rightOperand);
        };
    }

    private static OpMapper unaryOp(OpType op) {
        return (vertex, lookup, graphBuilder) -> {
            VertexUnaryOp unaryOpVertex = (VertexUnaryOp) vertex;
            Output<?> operand = getOutput(lookup, unaryOpVertex.getInput());
            return graphBuilder.unaryOp(op, getTensorflowOpName(vertex), operand);
        };
    }

    private static Output<?> createDoubleIf(Vertex<?> vertex, Map<VariableReference, Output<?>> lookup, TensorflowOpHelper graphBuilder) {
        DoubleIfVertex doubleIfVertex = (DoubleIfVertex) vertex;

        Output<Boolean> predicate = (Output<Boolean>) getOutput(lookup, doubleIfVertex.getPredicate());
        Output<Double> thn = (Output<Double>) getOutput(lookup, doubleIfVertex.getThn());
        Output<Double> els = (Output<Double>) getOutput(lookup, doubleIfVertex.getEls());

        long[] predicateShape = doubleIfVertex.getPredicate().getShape();
        Output<Long> shape = graphBuilder.constant(predicateShape, new long[]{predicateShape.length});

        Output<Double> thnBroadcast = graphBuilder.broadcastTo(thn, shape);
        Output<Double> elsBroadcast = graphBuilder.broadcastTo(els, shape);

        return graphBuilder.where(predicate, thnBroadcast, elsBroadcast);
    }

    private static Output<?> createConcat(Vertex<?> vertex, Map<VariableReference, Output<?>> lookup, TensorflowOpHelper graphBuilder) {
        ConcatenationVertex concatenationVertex = (ConcatenationVertex) vertex;

        Output<Double>[] inputs = (Output<Double>[]) Operands.asOutputs(
            Arrays.stream(concatenationVertex.getOperands())
                .map(v -> getOutput(lookup, v))
                .collect(Collectors.toList())
        );

        return graphBuilder.concat(inputs, concatenationVertex.getDimension(), getTensorflowOpName(concatenationVertex));
    }

    private static <T> Output<T> createSum(Vertex<?> vertex, Map<VariableReference, Output<?>> lookup, TensorflowOpHelper graphBuilder) {
        SumVertex summationVertex = (SumVertex) vertex;
        Output<?> input = getOutput(lookup, summationVertex.getInput());
        String name = getTensorflowOpName(vertex);

        int[] summingOverDimensions = summationVertex.getOverDimensions();
        Output<Integer> overDimensions;

        if (summingOverDimensions == null) {
            int inputRank = summationVertex.getInput().getShape().length;
            overDimensions = graphBuilder.constant(TensorShape.dimensionRange(0, inputRank), new long[]{inputRank});
        } else {
            int dims = summationVertex.getOverDimensions().length;
            overDimensions = graphBuilder.constant(summationVertex.getOverDimensions(), new long[]{dims});
        }

        return graphBuilder.binaryOp(OpType.SUM, name, input, overDimensions);
    }

    public static Output<?> createConstant(Vertex<?> vertex, TensorflowOpHelper graphBuilder) {

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

        throw new IllegalArgumentException("Cannot create " + value.getClass());
    }

    public static Output<?> createVariable(Vertex<?> vertex, TensorflowOpHelper graphBuilder) {

        Object value = vertex.getValue();

        if (value instanceof DoubleTensor) {
            return graphBuilder.variable(getTensorflowOpName(vertex), toShape(vertex.getShape()), Double.class);
        } else if (value instanceof IntegerTensor) {
            return graphBuilder.variable(getTensorflowOpName(vertex), toShape(vertex.getShape()), Integer.class);
        } else if (value instanceof BooleanTensor) {
            return graphBuilder.variable(getTensorflowOpName(vertex), toShape(vertex.getShape()), Boolean.class);
        }

        throw new IllegalArgumentException("Cannot create variable for " + value.getClass());
    }

    private static Shape toShape(long[] shape) {
        if (shape.length == 0) {
            return Shape.scalar();
        } else {
            return Shape.make(shape[0], Arrays.copyOfRange(shape, 1, shape.length));
        }
    }

    private static String getTensorflowOpName(Vertex vertex) {
        return vertex.getReference().toStringReference();
    }

}
