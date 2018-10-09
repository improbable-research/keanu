package io.improbable.keanu.backend.tensorflow;

import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.ADD;
import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.CONSTANT;
import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.DIVIDE;
import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.MATRIX_MULTIPLY;
import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.MULTIPLY;
import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.PLACE_HOLDER;
import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.POW;
import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.SUBTRACT;

import java.nio.DoubleBuffer;
import java.nio.IntBuffer;

import org.tensorflow.DataType;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;

import io.improbable.keanu.tensor.TensorShape;

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
        CONSTANT("Const"),
        PLACE_HOLDER("Placeholder"),
        NO_OP("NoOp"),
        DIVIDE("Div"),
        MULTIPLY("Mul"),
        SUBTRACT("Sub"),
        ADD("Add"),
        LOG("Log"),
        SUM("Sum"),
        POW("Pow"),
        MATRIX_MULTIPLY("MatMul");

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

    <T> Output<T> div(Output<T> x, Output<T> y, String name) {
        return binaryOp(DIVIDE, name, x, y);
    }

    <T> Output<T> sub(Output<T> left, Output<T> right, String name) {
        return binaryOp(SUBTRACT, name, left, right);
    }

    <T> Output<T> mul(Output<T> left, Output<T> right, String name) {
        return binaryOp(MULTIPLY, name, left, right);
    }

    <T> Output<T> add(Output<T> left, Output<T> right, String name) {
        return binaryOp(ADD, name, left, right);
    }

    <T> Output<T> mmul(Output<T> left, Output<T> right, String name) {
        return binaryOp(MATRIX_MULTIPLY, name, left, right);
    }

    <T> Output<T> pow(Output<T> base, Output<T> power, String name) {
        return binaryOp(POW, name, base, power);
    }

    <T> Output<T> log(Output<T> input, String name) {
        return unaryOp(OpType.LOG, name, input);
    }

    <T> Output<T> reduceSum(Output<T> input, String name) {

        int dims = input.shape().numDimensions();
        Output<Integer> dimRange = constant(TensorShape.dimensionRange(0, dims), new long[]{dims}, name + "_dimRange");

        return binaryOp(OpType.SUM, name, input, dimRange);
    }

    <T> Output<T> concat(Output<T>[] inputs, int dimension, String name) {
        Output<Integer> dim = constant(dimension, name + "_dim");

        OperationBuilder opBuilder = scope.graph().opBuilder("ConcatV2", name);
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

    private <T, L, R> Output<T> binaryOp(OpType type, String name, Output<L> in1, Output<R> in2) {
        return scope.graph().opBuilder(type.opName, name).addInput(in1).addInput(in2).build().output(0);
    }

    private <T> Output<T> unaryOp(OpType type, String name, Output<T> in1) {
        return scope.graph().opBuilder(type.opName, name).addInput(in1).build().output(0);
    }
}