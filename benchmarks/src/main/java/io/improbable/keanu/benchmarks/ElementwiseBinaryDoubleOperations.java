package io.improbable.keanu.benchmarks;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;

@State(Scope.Benchmark)
public class ElementwiseBinaryDoubleOperations {

    private static final int NUM_OPERATIONS = 100;

    @Param({"TIMES", "MATRIX_MULTIPLY"})
    public BinaryOperation operation;

    @Param({"1", "10", "100", "1000"})
    public int dimLength;

    @Param({"JVM", "ND4J"})
    public DoubleTensorImpl impl;

    DoubleTensor left;
    DoubleTensor right;

    @Setup
    public void setup() {
        DoubleTensor.setFactory(impl.getFactory());
        left = DoubleTensor.arange(0, dimLength * dimLength).reshape(dimLength, dimLength);
        right = DoubleTensor.arange(dimLength * dimLength, 2 * dimLength * dimLength).reshape(dimLength, dimLength);
    }

    @Benchmark
    public DoubleTensor benchmark() {

        DoubleTensor result = null;
        for (int i = 0; i < NUM_OPERATIONS; i++) {
            result = operation.apply(left, right);
        }

        return result;
    }
}
