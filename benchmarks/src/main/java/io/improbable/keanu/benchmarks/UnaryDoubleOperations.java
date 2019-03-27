package io.improbable.keanu.benchmarks;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;

@State(Scope.Benchmark)
public class UnaryDoubleOperations {

    private static final int NUM_OPERATIONS = 100;

    @Param({"CHOLESKY", "DETERMINANT", "COS", "INVERSE", "LOG", "TRANSPOSE"})
    public UnaryOperation operation;

    @Param({"10"})
    public int dimLength;

    @Param({"JVM", "ND4J"})
    public DoubleTensorImpl impl;

    DoubleTensor tensor;

    @Setup
    public void setup() {
        DoubleTensor.setFactory(impl.getFactory());
        KeanuRandom random = new KeanuRandom(1);
        DoubleTensor A = random.nextDouble(new long[]{dimLength, dimLength});

        /*
         * Generate a matrix that is symmetric with values in a range suitable for all the operations above
         */
        tensor = A.matrixMultiply(A.transpose()).times(0.5);
    }

    @Benchmark
    public DoubleTensor benchmark() {

        DoubleTensor result = null;
        for (int i = 0; i < NUM_OPERATIONS; i++) {
            result = operation.apply(tensor);
        }

        return result;
    }
}
