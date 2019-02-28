package io.improbable.keanu.benchmarks;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensorFactory;
import io.improbable.keanu.tensor.dbl.JVMDoubleTensorFactory;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensorFactory;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;

@State(Scope.Benchmark)
public class ElementwiseDoubleOperations {

    private static final int NUM_OPERATIONS = 100;

    @Param({"TIMES", "MATRIX_MULTIPLY"})
    public Operation operation;

    @Param({"1", "10", "100", "1000"})
    public int N;

    public enum DoubleTensorImpl {
        JVM {
            @Override
            DoubleTensorFactory getFactory() {
                return new JVMDoubleTensorFactory();
            }
        },
        ND4J {
            @Override
            DoubleTensorFactory getFactory() {
                return new Nd4jDoubleTensorFactory();
            }
        };

        abstract DoubleTensorFactory getFactory();
    }

    @Param({"JVM", "ND4J"})
    public DoubleTensorImpl impl;

    DoubleTensor left;
    DoubleTensor right;

    @Setup
    public void setup() {
        DoubleTensor.setFactory(impl.getFactory());
        left = DoubleTensor.arange(0, N * N).reshape(N, N);
        right = DoubleTensor.arange(N * N, 2 * N * N).reshape(N, N);
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
