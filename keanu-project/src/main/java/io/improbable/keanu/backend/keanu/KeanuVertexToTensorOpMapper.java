package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexBinaryOp;
import io.improbable.keanu.vertices.VertexUnaryOp;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.*;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.*;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

public class KeanuVertexToTensorOpMapper {

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

        opMappers.put(SinVertex.class, unaryOp("sin"));
        opMappers.put(CosVertex.class, unaryOp("cos"));
        opMappers.put(TanVertex.class, unaryOp("tan"));

        opMappers.put(ExpVertex.class, unaryOp("exp"));

        opMappers.put(LogVertex.class, unaryOp("log"));
        opMappers.put(LogGammaVertex.class, unaryOp("logGamma"));

        opMappers.put(SumVertex.class, KeanuVertexToTensorOpMapper::sumOp);
    }

    interface OpMapper {
        String apply(Vertex<?> vertex, Map<VariableReference, String> lookup);
    }

    public static OpMapper getOpMapperFor(Class<?> clazz) {
        return opMappers.get(clazz);
    }

    private static OpMapper binaryOp(String methodName) {
        return (vertex, lookup) -> {
            VertexBinaryOp<?, ?> binaryOpVertex = (VertexBinaryOp<?, ?>) vertex;
            Vertex<?> left = binaryOpVertex.getLeft();
            Vertex<?> right = binaryOpVertex.getRight();
            return lookup.get(left.getReference()) + "." + methodName + "(" + lookup.get(right.getReference()) + ")";
        };
    }

    private static OpMapper unaryOp(String methodName) {
        return (vertex, lookup) -> {
            VertexUnaryOp unaryOpVertex = (VertexUnaryOp) vertex;
            Vertex<?> input = unaryOpVertex.getInput();
            return lookup.get(input.getReference()) + "." + methodName + "()";
        };
    }

    private static String sumOp(Vertex<?> vertex, Map<VariableReference, String> lookup) {
        SumVertex sumVertex = (SumVertex) vertex;

        int[] dimensions = sumVertex.getOverDimensions();

        String declaration = lookup.get(sumVertex.getInput().getReference());

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
