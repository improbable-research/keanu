package io.improbable.keanu.benchmarks;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
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
        final Nd4jDoubleTensor value = Nd4jDoubleTensor.scalar(1.);
        final Nd4jDoubleTensor operand = Nd4jDoubleTensor.scalar(OPERAND);
        DoubleTensor result = value;

        for (int i = 0; i < NUM_OPERATIONS; i++) {
            result = operation.apply(result, operand);
        }
        return result.scalar();
    }

    @Benchmark
    public double customScalarClass() {
        final ScalarDoubleTensor value = new ScalarDoubleTensor(1.);
        final ScalarDoubleTensor operand = new ScalarDoubleTensor(OPERAND);
        DoubleTensor result = value;

        for (int i = 0; i < NUM_OPERATIONS; i++) {
            result = operation.apply(result, operand);
        }
        return result.scalar();
    }
}
