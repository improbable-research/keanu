package io.improbable.keanu.benchmarks;

import io.improbable.keanu.tensor.TensorFactories;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import static java.util.concurrent.TimeUnit.MILLISECONDS;

@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1000, timeUnit = MILLISECONDS)
@Measurement(iterations = 5, time = 1000, timeUnit = MILLISECONDS)
@Fork(3)
public class ElementwiseBinaryDoubleOperations {

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
        TensorFactories.doubleTensorFactory = impl.getFactory();
        left = DoubleTensor.arange(0, dimLength * dimLength).reshape(dimLength, dimLength);
        right = DoubleTensor.arange(dimLength * dimLength, 2 * dimLength * dimLength).reshape(dimLength, dimLength);
    }

    @Benchmark
    public DoubleTensor benchmark() {

        DoubleTensor result = operation.apply(left, right);

        return result;
    }
}
