package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexBinaryOp;
import io.improbable.keanu.vertices.VertexUnaryOp;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DifferenceVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DivisionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;

import java.util.HashMap;
import java.util.Map;

public class KeanuVertexToTensorOpMapper {

    private static Map<Class<?>, OpMapper> opMappers;

    static {
        opMappers = new HashMap<>();

        //double binary ops
        opMappers.put(AdditionVertex.class, binaryOp("plus"));
        opMappers.put(DifferenceVertex.class, binaryOp("minus"));
        opMappers.put(DivisionVertex.class, binaryOp("div"));
        opMappers.put(MultiplicationVertex.class, binaryOp("times"));

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

}
