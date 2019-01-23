package io.improbable.keanu.backend.keanu.compiled;

import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexBinaryOp;
import io.improbable.keanu.vertices.VertexUnaryOp;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.*;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.*;

import java.util.*;
import java.util.stream.Collectors;

public class KeanuVertexToTensorOpMapper {

    public static final boolean ENABLE_IN_PLACE = true;

    private static Map<Class<?>, OpMapper> opMappers;

    static {
        opMappers = new HashMap<>();

        //double binary ops
        opMappers.put(AdditionVertex.class, binaryOp("plus"));
        opMappers.put(DifferenceVertex.class, binaryOp("minus"));
        opMappers.put(DivisionVertex.class, binaryOp("div"));
        opMappers.put(MultiplicationVertex.class, binaryOp("times"));
        opMappers.put(MatrixMultiplicationVertex.class, binaryOp("matrixMultiply"));
        opMappers.put(PowerVertex.class, binaryOp("pow"));

        opMappers.put(AbsVertex.class, unaryOp("abs"));
        opMappers.put(CosVertex.class, unaryOp("cos"));
        opMappers.put(ExpVertex.class, unaryOp("exp"));
        opMappers.put(LogVertex.class, unaryOp("log"));
        opMappers.put(LogGammaVertex.class, unaryOp("logGamma"));
        opMappers.put(SinVertex.class, unaryOp("sin"));
        opMappers.put(TanVertex.class, unaryOp("tan"));

        opMappers.put(SumVertex.class, KeanuVertexToTensorOpMapper::sumOp);
    }

    interface OpMapper {
        String apply(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup);
    }

    public static OpMapper getOpMapperFor(Class<?> clazz) {
        return opMappers.get(clazz);
    }

    private static OpMapper binaryOp(String methodName) {
        return (vertex, lookup) -> {
            VertexBinaryOp<?, ?> binaryOpVertex = (VertexBinaryOp<?, ?>) vertex;
            Vertex<?> left = binaryOpVertex.getLeft();
            Vertex<?> right = binaryOpVertex.getRight();

            KeanuCompiledVariable leftVariable = lookup.get(left.getReference());
            KeanuCompiledVariable rightVariable = lookup.get(right.getReference());
            boolean doInPlace = leftVariable.isMutable() && isLastChildByToposort(vertex, left) && ENABLE_IN_PLACE;
            String call = doInPlace ? methodName + "InPlace" : methodName;

            return leftVariable.getName() + "." + call + "(" + rightVariable.getName() + ")";

        };
    }

    private static OpMapper unaryOp(String methodName) {
        return (vertex, lookup) -> {
            VertexUnaryOp unaryOpVertex = (VertexUnaryOp) vertex;
            Vertex<?> input = unaryOpVertex.getInput();

            KeanuCompiledVariable inputVariable = lookup.get(input.getReference());
            boolean doInPlace = inputVariable.isMutable() && isLastChildByToposort(vertex, input) && ENABLE_IN_PLACE;

            String call = doInPlace ? methodName + "InPlace" : methodName;

            return inputVariable.getName() + "." + call + "()";
        };
    }

    private static boolean isLastChildByToposort(Vertex<?> child, Vertex<?> parent) {
        Optional<Vertex> last = parent.getChildren().stream()
            .max(Comparator.comparing(Vertex::getId));

        return last
            .map(v -> v.getId().equals(child.getId()))
            .orElse(false);
    }

    private static String sumOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        SumVertex sumVertex = (SumVertex) vertex;

        int[] dimensions = sumVertex.getOverDimensions();

        String declaration = lookup.get(sumVertex.getInput().getReference()).getName();

        if (dimensions != null) {
            String dims = Arrays.stream(dimensions)
                .mapToObj(i -> i + "")
                .collect(Collectors.joining(","));

            String args = "new int[]{" + dims + "}";
            return declaration + ".sum(" + args + ")";
        } else {
            return "DoubleTensor.scalar(" + declaration + ".sum())";
        }

    }

}
