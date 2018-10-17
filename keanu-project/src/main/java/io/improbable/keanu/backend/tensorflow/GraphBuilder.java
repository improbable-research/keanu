package io.improbable.keanu.backend.tensorflow;

import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.ADD;
import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.CONCAT_V2;
import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.CONSTANT;
import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.PLACE_HOLDER;

import java.nio.DoubleBuffer;
import java.nio.IntBuffer;

import org.tensorflow.DataType;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;

public class GraphBuilder {

    private Scope scope;

    public GraphBuilder(Scope scope) {
        this.scope = scope;
    }

    public <T> Output<T> getOutput(String name) {

        Operation operation = scope.graph().operation(name);
        if (operation == null) {
            return null;
        }
        return operation.output(0);
    }

    public enum OpType {
        //Number ops
        ADD("Add"),
        ABS("Abs"),
        ATAN2("Atan2"),
        ACOS("Acos"),
        ASIN("Asin"),
        ATAN("Atan"),
        CEIL("Ceil"),
        COS("Cos"),
        DIVIDE("Div"),
        EXP("Exp"),
        FLOOR("Floor"),
        LOG("Log"),
        LOG_GAMMA("Lgamma"),
        MATRIX_MULTIPLY("MatMul"),
        MAX("Max"),
        MIN("Min"),
        MULTIPLY("Mul"),
        MATRIX_DETERMINANT("MatrixDeterminant"),
        MATRIX_INVERSE("MatrixInverse"),
        POW("Pow"),
        RESHAPE("Reshape"),
        ROUND("Round"),
        SUBTRACT("Sub"),
        SIGMOID("Sigmoid"),
        SIN("Sin"),
        SUM("Sum"),
        TAN("Tan"),

        //Boolean
        AND("LogicalAnd"),
        OR("LogicalOr"),
        NOT("LogicalNot"),

        //Generic
        CONCAT_V2("ConcatV2"),
        CONSTANT("Const"),
        PLACE_HOLDER("Placeholder"),
        NO_OP("NoOp");

        public final String opName;

        OpType(String opName) {
            this.opName = opName;
        }
    }

    public enum AttrName {
        DTYPE("dtype"),
        VALUE("value"),
        SHAPE("shape");

        public final String attrName;

        AttrName(String attrName) {
            this.attrName = attrName;
        }
    }

    <T> Output<T> add(Output<T> left, Output<T> right, String name) {
        return binaryOp(ADD, name, left, right);
    }

    <T> Output<T> concat(Output<T>[] inputs, int dimension, String name) {
        Output<Integer> dim = constant(dimension, name + "_dim");

        OperationBuilder opBuilder = scope.graph().opBuilder(CONCAT_V2.opName, name);
        opBuilder.addInputList(inputs);
        opBuilder.addInput(dim.asOutput());

        return opBuilder.build().output(0);
    }

    Output<Double> constant(double value, String name) {
        try (Tensor<Double> tensor = Tensor.create(value, Double.class)) {
            return this.constant(name, tensor, Double.class);
        }
    }

    Output<Double> constant(double[] value, long[] shape, String name) {
        try (Tensor<Double> tensor = Tensor.create(shape, DoubleBuffer.wrap(value))) {
            return this.constant(name, tensor, Double.class);
        }
    }

    Output<Integer> constant(int[] value, long[] shape, String name) {
        try (Tensor<Integer> tensor = Tensor.create(shape, IntBuffer.wrap(value))) {
            return this.constant(name, tensor, Integer.class);
        }
    }

    Output<Integer> constant(int value, String name) {
        try (Tensor<Integer> tensor = Tensor.create(value, Integer.class)) {
            return this.constant(name, tensor, Integer.class);
        }
    }

    private <T> Output<T> constant(String name, Tensor<T> tensor, Class<T> type) {
        return scope.graph().opBuilder(CONSTANT.opName, name)
            .setAttr(AttrName.DTYPE.attrName, DataType.fromClass(type))
            .setAttr(AttrName.VALUE.attrName, tensor)
            .build()
            .output(0);
    }

    public <T> Output<T> placeholder(String name, Shape shape, Class<T> type) {
        return scope.graph().opBuilder(PLACE_HOLDER.opName, name)
            .setAttr(AttrName.DTYPE.attrName, DataType.fromClass(type))
            .setAttr(AttrName.SHAPE.attrName, shape)
            .build()
            .output(0);
    }

    public <T, L, R> Output<T> binaryOp(OpType type, String name, Output<L> in1, Output<R> in2) {
        return scope.graph().opBuilder(type.opName, name)
            .addInput(in1)
            .addInput(in2).build()
            .output(0);
    }

    public <T> Output<T> unaryOp(OpType type, String name, Output<T> in1) {
        return scope.graph().opBuilder(type.opName, name)
            .addInput(in1).build()
            .output(0);
    }
}