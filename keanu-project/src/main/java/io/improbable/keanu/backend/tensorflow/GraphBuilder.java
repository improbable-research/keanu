package io.improbable.keanu.backend.tensorflow;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.BroadcastTo;
import org.tensorflow.op.core.Where3;

import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.ADD;
import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.CONCAT_V2;
import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.CONSTANT;
import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.PLACE_HOLDER;
import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.VARIABLE_V2;

public class GraphBuilder {

    private Scope scope;

    public GraphBuilder(Scope scope) {
        this.scope = scope;
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
        VARIABLE_V2("VariableV2"),
        ASSIGN("Assign"),
        NO_OP("NoOp");

        public final String tfOpName;

        OpType(String tfOpName) {
            this.tfOpName = tfOpName;
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


    <T> Output<T> add(Output<T> left, Output<T> right) {
        return binaryOp(ADD, scope.makeOpName(ADD.tfOpName), left, right);
    }

    <T> Output<T> add(Output<T> left, Output<T> right, String name) {
        return binaryOp(ADD, name, left, right);
    }

    <T> Output<T> concat(Output<T>[] inputs, int dimension, String name) {
        Output<Integer> dim = constant(dimension, name + "_dim");

        OperationBuilder opBuilder = scope.graph().opBuilder(CONCAT_V2.tfOpName, name);
        opBuilder.addInputList(inputs);
        opBuilder.addInput(dim.asOutput());

        return opBuilder.build().output(0);
    }

    <T> Output<T> where(Output<Boolean> predicate, Output<T> thn, Output<T> els) {
        return Where3.create(scope, predicate, thn, els).asOutput();
    }

    <T, U extends Number> Output<T> broadcastTo(Output<T> input, Output<U> shape) {
        return BroadcastTo.create(scope, input, shape).asOutput();
    }

    Output<Double> constant(double value) {
        return constant(value, scope.makeOpName(CONSTANT.tfOpName));
    }

    Output<Double> constant(double value, String name) {
        try (Tensor<Double> tensor = Tensor.create(value, Double.class)) {
            return this.constant(name, tensor, Double.class);
        }
    }

    Output<Double> constant(double[] value, long[] shape) {
        return constant(value, shape, scope.makeOpName(CONSTANT.tfOpName));
    }

    Output<Double> constant(double[] value, long[] shape, String name) {
        try (Tensor<Double> tensor = Tensor.create(shape, DoubleBuffer.wrap(value))) {
            return this.constant(name, tensor, Double.class);
        }
    }

    Output<Boolean> constant(Boolean[] value, long[] shape) {
        return constant(value, shape, scope.makeOpName(CONSTANT.tfOpName));
    }

    Output<Boolean> constant(Boolean[] value, long[] shape, String name) {
        try (Tensor<Boolean> tensor = TensorflowData.toTensorFlow(shape, value)) {
            return this.constant(name, tensor, Boolean.class);
        }
    }

    Output<Integer> constant(int[] value, long[] shape) {
        return constant(value, shape, scope.makeOpName(CONSTANT.tfOpName));
    }

    Output<Integer> constant(int[] value, long[] shape, String name) {
        try (Tensor<Integer> tensor = Tensor.create(shape, IntBuffer.wrap(value))) {
            return this.constant(name, tensor, Integer.class);
        }
    }

    Output<Long> constant(long[] value, long[] shape) {
        return constant(value, shape, scope.makeOpName(CONSTANT.tfOpName));
    }

    Output<Long> constant(long[] value, long[] shape, String name) {
        try (Tensor<Long> tensor = Tensor.create(shape, LongBuffer.wrap(value))) {
            return this.constant(name, tensor, Long.class);
        }
    }

    Output<Integer> constant(int value) {
        return constant(value, scope.makeOpName(CONSTANT.tfOpName));
    }

    Output<Integer> constant(int value, String name) {
        try (Tensor<Integer> tensor = Tensor.create(value, Integer.class)) {
            return this.constant(name, tensor, Integer.class);
        }
    }

    private <T> Output<T> constant(String name, Tensor<T> tensor, Class<T> type) {
        return scope.graph().opBuilder(CONSTANT.tfOpName, name)
            .setAttr(AttrName.DTYPE.attrName, DataType.fromClass(type))
            .setAttr(AttrName.VALUE.attrName, tensor)
            .build()
            .output(0);
    }

    public <T> Output<T> variable(String name, Shape shape, Class<T> type) {
        OperationBuilder opBuilder = scope.graph().opBuilder(VARIABLE_V2.tfOpName, name);
        opBuilder.setAttr(AttrName.SHAPE.attrName, shape);
        opBuilder.setAttr(AttrName.DTYPE.attrName, DataType.fromClass(type));
        return opBuilder.build().output(0);
    }

    public <T> Output<T> assign(Operand<T> ref, Operand<T> value) {
        OperationBuilder opBuilder = scope.graph().opBuilder("Assign", scope.makeOpName("Assign"));
        opBuilder.addInput(ref.asOutput());
        opBuilder.addInput(value.asOutput());
        return opBuilder.build().output(0);
    }

    public <T> Output<T> placeholder(String name, Shape shape, Class<T> type) {
        return scope.graph().opBuilder(PLACE_HOLDER.tfOpName, name)
            .setAttr(AttrName.DTYPE.attrName, DataType.fromClass(type))
            .setAttr(AttrName.SHAPE.attrName, shape)
            .build()
            .output(0);
    }

    public <T, L, R> Output<T> binaryOp(OpType type, String name, Output<L> in1, Output<R> in2) {
        return scope.graph().opBuilder(type.tfOpName, name)
            .addInput(in1)
            .addInput(in2).build()
            .output(0);
    }

    public <T> Output<T> unaryOp(OpType type, String name, Output<T> in1) {
        return scope.graph().opBuilder(type.tfOpName, name)
            .addInput(in1).build()
            .output(0);
    }
}