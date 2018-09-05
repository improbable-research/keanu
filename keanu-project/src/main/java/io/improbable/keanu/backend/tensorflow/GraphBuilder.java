package io.improbable.keanu.backend.tensorflow;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

// In the fullness of time, equivalents of the methods of this class should be auto-generated from
// the OpDefs linked into libtensorflow_jni.so. That would match what is done in other languages
// like Python, C++ and Go.
public class GraphBuilder {

    private Graph g;

    public GraphBuilder(Graph g) {
        this.g = g;
    }

    <T> Output<T> div(Output<T> x, Output<T> y) {
        return binaryOp("Div", x, y);
    }

    <T> Output<T> sub(Output<T> x, Output<T> y) {
        return binaryOp("Sub", x, y);
    }

    <T, U> Output<U> cast(Output<T> value, Class<U> type) {
        DataType dtype = DataType.fromClass(type);
        return g.opBuilder("Cast", "Cast")
            .addInput(value)
            .setAttr("DstT", dtype)
            .build()
            .<U>output(0);
    }

    <T> Output<T> constant(String name, Object value, Class<T> type) {
        try (Tensor<T> t = Tensor.<T>create(value, type)) {
            return g.opBuilder("Const", name)
                .setAttr("dtype", DataType.fromClass(type))
                .setAttr("value", t)
                .build()
                .<T>output(0);
        }
    }

    Output<String> constant(String name, byte[] value) {
        return this.constant(name, value, String.class);
    }

    Output<Integer> constant(String name, int value) {
        return this.constant(name, value, Integer.class);
    }

    Output<Integer> constant(String name, int[] value) {
        return this.constant(name, value, Integer.class);
    }

    Output<Double> constant(String name, double value) {
        return this.constant(name, value, Double.class);
    }

    private <T> Output<T> binaryOp(String type, Output<T> in1, Output<T> in2) {
        return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T>output(0);
    }
}