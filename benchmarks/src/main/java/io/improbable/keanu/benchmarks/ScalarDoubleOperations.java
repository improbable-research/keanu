package io.improbable.keanu.benchmarks;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.ScalarDoubleTensor;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;

@State(Scope.Benchmark)
public class ScalarDoubleOperations {


    private static final double OPERAND = 1.00001;
    private static final int NUM_OPERATIONS = 1000000;

    @Param({"PLUS", "MINUS", "TIMES", "DIVIDE"})
    public Operation operation;

    @Benchmark
    public double baseline() {
        double value = 1.;
        for (int i = 0; i < NUM_OPERATIONS; i++) {
            value = operation.apply(value, OPERAND);
        }
        return value;
    }

    @Benchmark
    public double nd4jScalars() {
        DoubleTensor value = DoubleTensor.scalar(1.);
        DoubleTensor operand = DoubleTensor.scalar(OPERAND);

        for (int i = 0; i < NUM_OPERATIONS; i++) {
            value = operation.apply(value, operand);
        }
        return value.scalar();
    }

    @Benchmark
    public double customScalarClass() {
        DoubleTensor value = new ScalarDoubleTensor(1.);
        DoubleTensor operand = new ScalarDoubleTensor(OPERAND);

        for (int i = 0; i < NUM_OPERATIONS; i++) {
            value = operation.apply(value, operand);
        }
        return value.scalar();
    }
}
