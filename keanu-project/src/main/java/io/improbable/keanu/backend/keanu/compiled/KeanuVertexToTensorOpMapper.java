package io.improbable.keanu.backend.keanu.compiled;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexBinaryOp;
import io.improbable.keanu.vertices.VertexUnaryOp;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.CastToDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
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
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SigmoidVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SinVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SumVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.TanVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.CastToIntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerAdditionVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerDifferenceVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerDivisionVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMaxVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMinVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMultiplicationVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerPowerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.multiple.IntegerConcatenationVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerAbsVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerSumVertex;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

public class KeanuVertexToTensorOpMapper {

    public static final boolean ENABLE_IN_PLACE = true;

    private static Map<Class<?>, OpMapper> opMappers;

    static {
        opMappers = new HashMap<>();

        //Double ops
        opMappers.put(AdditionVertex.class, fluentBinaryOp("plus"));
        opMappers.put(DifferenceVertex.class, fluentBinaryOp("minus"));
        opMappers.put(DivisionVertex.class, fluentBinaryOp("div"));
        opMappers.put(MultiplicationVertex.class, fluentBinaryOp("times"));
        opMappers.put(MatrixMultiplicationVertex.class, fluentBinaryOp("matrixMultiply"));
        opMappers.put(PowerVertex.class, fluentBinaryOp("pow"));

        opMappers.put(AbsVertex.class, unaryOp("abs"));
        opMappers.put(CosVertex.class, unaryOp("cos"));
        opMappers.put(ArcCosVertex.class, unaryOp("acos"));
        opMappers.put(ExpVertex.class, unaryOp("exp"));
        opMappers.put(LogVertex.class, unaryOp("log"));
        opMappers.put(LogGammaVertex.class, unaryOp("logGamma"));
        opMappers.put(SinVertex.class, unaryOp("sin"));
        opMappers.put(ArcSinVertex.class, unaryOp("asin"));
        opMappers.put(TanVertex.class, unaryOp("tan"));
        opMappers.put(ArcTanVertex.class, unaryOp("atan"));
        opMappers.put(MatrixDeterminantVertex.class, unaryOp("determinant"));
        opMappers.put(CeilVertex.class, unaryOp("ceil"));
        opMappers.put(FloorVertex.class, unaryOp("floor"));
        opMappers.put(RoundVertex.class, unaryOp("round"));
        opMappers.put(SigmoidVertex.class, unaryOp("sigmoid"));
        opMappers.put(MatrixInverseVertex.class, unaryOp("inverse"));

        opMappers.put(ConcatenationVertex.class, KeanuVertexToTensorOpMapper::concatOpDouble);
        opMappers.put(SumVertex.class, KeanuVertexToTensorOpMapper::sumDoubleOp);
        opMappers.put(MaxVertex.class, binaryOp("DoubleTensor.max(%s,%s)"));
        opMappers.put(MinVertex.class, binaryOp("DoubleTensor.min(%s,%s)"));

        opMappers.put(CastToDoubleVertex.class, unaryOp("toDouble"));

        //Integer ops
        opMappers.put(IntegerMultiplicationVertex.class, fluentBinaryOp("times"));
        opMappers.put(IntegerAdditionVertex.class, fluentBinaryOp("plus"));
        opMappers.put(IntegerDifferenceVertex.class, fluentBinaryOp("minus"));
        opMappers.put(IntegerDivisionVertex.class, fluentBinaryOp("divideBy"));
        opMappers.put(IntegerAbsVertex.class, fluentBinaryOp("abs"));
        opMappers.put(IntegerPowerVertex.class, fluentBinaryOp("pow"));

        opMappers.put(IntegerConcatenationVertex.class, KeanuVertexToTensorOpMapper::concatOpInteger);
        opMappers.put(IntegerSumVertex.class, KeanuVertexToTensorOpMapper::sumIntegerOp);
        opMappers.put(IntegerMaxVertex.class, binaryOp("IntegerTensor.max(%s,%s)"));
        opMappers.put(IntegerMinVertex.class, binaryOp("IntegerTensor.min(%s,%s)"));

        opMappers.put(CastToIntegerVertex.class, unaryOp("toInteger"));

        //Constants
        opMappers.put(ConstantIntegerVertex.class, KeanuVertexToTensorOpMapper::constant);
        opMappers.put(ConstantDoubleVertex.class, KeanuVertexToTensorOpMapper::constant);
        opMappers.put(ConstantBooleanVertex.class, KeanuVertexToTensorOpMapper::constant);
    }

    interface OpMapper {
        String apply(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup);
    }

    public static OpMapper getOpMapperFor(Class<?> clazz) {
        return opMappers.get(clazz);
    }

    private static OpMapper fluentBinaryOp(String methodName) {
        return (vertex, lookup) -> {
            VertexBinaryOp<?, ?> binaryOpVertex = (VertexBinaryOp<?, ?>) vertex;
            Vertex<?> left = binaryOpVertex.getLeft();
            Vertex<?> right = binaryOpVertex.getRight();

            KeanuCompiledVariable leftVariable = lookup.get(left.getReference());
            KeanuCompiledVariable rightVariable = lookup.get(right.getReference());
            boolean doInPlace = leftVariable.isMutable() && isLastChildByTopographicalSort(vertex, left) && ENABLE_IN_PLACE;
            String call = doInPlace ? methodName + "InPlace" : methodName;

            return leftVariable.getName() + "." + call + "(" + rightVariable.getName() + ")";

        };
    }

    private static OpMapper unaryOp(String methodName) {
        return (vertex, lookup) -> {
            VertexUnaryOp unaryOpVertex = (VertexUnaryOp) vertex;
            Vertex<?> input = unaryOpVertex.getInput();

            KeanuCompiledVariable inputVariable = lookup.get(input.getReference());
            boolean doInPlace = inputVariable.isMutable() && isLastChildByTopographicalSort(vertex, input) && ENABLE_IN_PLACE;

            String call = doInPlace ? methodName + "InPlace" : methodName;

            return inputVariable.getName() + "." + call + "()";
        };
    }

    private static boolean isLastChildByTopographicalSort(Vertex<?> child, Vertex<?> parent) {
        Optional<Vertex> last = parent.getChildren().stream()
            .max(Comparator.comparing(Vertex::getId));

        return last
            .map(v -> v.getId().equals(child.getId()))
            .orElse(false);
    }

    private static String constant(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        throw new IllegalArgumentException("Constant should not be operation mapped");
    }

    private static OpMapper binaryOp(String format) {
        return (vertex, lookup) -> {
            VertexBinaryOp<?, ?> binaryOpVertex = (VertexBinaryOp<?, ?>) vertex;
            Vertex<?> left = binaryOpVertex.getLeft();
            Vertex<?> right = binaryOpVertex.getRight();

            KeanuCompiledVariable leftVariable = lookup.get(left.getReference());
            KeanuCompiledVariable rightVariable = lookup.get(right.getReference());

            return String.format(format, leftVariable.getName(), rightVariable.getName());
        };
    }

    private static String concatOp(int dimension,
                                   Vertex[] operands,
                                   String concatOp,
                                   Map<VariableReference, KeanuCompiledVariable> lookup) {

        String operandArg = Arrays.stream(operands)
            .map(v -> lookup.get(v).getName())
            .collect(Collectors.joining(","));

        return concatOp + "(" + dimension + "," + operandArg + ")";

    }

    private static String concatOpDouble(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        ConcatenationVertex concatenationVertex = (ConcatenationVertex) vertex;
        DoubleVertex[] operands = concatenationVertex.getOperands();
        int dimension = concatenationVertex.getDimension();

        return concatOp(dimension, operands, "DoubleTensor.concat", lookup);
    }

    private static String concatOpInteger(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        IntegerConcatenationVertex concatenationVertex = (IntegerConcatenationVertex) vertex;
        IntegerVertex[] operands = concatenationVertex.getInputArray();
        int dimension = concatenationVertex.getDimension();

        return concatOp(dimension, operands, "IntegerTensor.concat", lookup);
    }

    private static String sumDoubleOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        SumVertex sumVertex = (SumVertex) vertex;

        int[] dimensions = sumVertex.getOverDimensions();

        return sumOp(
            sumVertex.getInput().getReference(),
            dimensions,
            "DoubleTensor.scalar",
            lookup
        );
    }

    private static String sumIntegerOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        IntegerSumVertex sumVertex = (IntegerSumVertex) vertex;

        int[] dimensions = sumVertex.getOverDimensions();

        return sumOp(
            sumVertex.getInput().getReference(),
            dimensions,
            "IntegerTensor.scalar",
            lookup
        );
    }

    private static String sumOp(VariableReference inputReference,
                                int[] dimensions,
                                String scalarFactory,
                                Map<VariableReference, KeanuCompiledVariable> lookup) {

        String declaration = lookup.get(inputReference).getName();

        if (dimensions != null) {
            String dims = Arrays.stream(dimensions)
                .mapToObj(i -> i + "")
                .collect(Collectors.joining(","));

            String args = "new int[]{" + dims + "}";
            return declaration + ".sum(" + args + ")";
        } else {
            return scalarFactory + "(" + declaration + ".sum())";
        }
    }

}
