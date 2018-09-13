package io.improbable.keanu.backend.tensorflow;

import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.ADD;
import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.CONSTANT;
import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.DIVIDE;
import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.MATRIX_MULTIPLY;
import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.MULTIPLY;
import static io.improbable.keanu.backend.tensorflow.GraphBuilder.OpType.SUBTRACT;

import java.nio.DoubleBuffer;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

public class GraphBuilder {

    private Graph g;

    public GraphBuilder(Graph g) {
        this.g = g;
    }

    public <T> Output<T> getOutput(String name) {
        return g.operation(name).output(0);
    }

    public enum OpType {
        CONSTANT("Const"),
        DIVIDE("Div"),
        MULTIPLY("Mul"),
        SUBTRACT("Sub"),
        ADD("Add"),
        MATRIX_MULTIPLY("MatMul");

        public final String name;

        OpType(String tensorFlowName) {
            this.name = tensorFlowName;
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

    private <T> Output<T> constant(String name, Tensor<T> tensor, Class<T> type) {
        return g.opBuilder(CONSTANT.name, name)
            .setAttr("dtype", DataType.fromClass(type))
            .setAttr("value", tensor)
            .build()
            .output(0);
    }

    private <T> Output<T> binaryOp(OpType type, Output<T> in1, Output<T> in2) {
        return g.opBuilder(type.name, type.name).addInput(in1).addInput(in2).build().output(0);
    }

    private <T> Output<T> binaryOp(OpType type, String name, Output<T> in1, Output<T> in2) {
        return g.opBuilder(type.name, name).addInput(in1).addInput(in2).build().output(0);
    }
}