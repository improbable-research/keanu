package io.improbable.keanu.benchmarks;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;

@State(Scope.Benchmark)
public class Nd4jDoubleTensorScalarOperations {

    public DoubleTensor tensor;

    public static final double SCALAR_VALUE = 42.;
    public static final DoubleTensor RANK_0_SCALAR_TENSOR = Nd4jDoubleTensor.scalar(SCALAR_VALUE);
    public static final DoubleTensor RANK_1_SCALAR_TENSOR = Nd4jDoubleTensor.scalar(SCALAR_VALUE).reshape(1);
    public static final DoubleTensor RANK_2_SCALAR_TENSOR = Nd4jDoubleTensor.scalar(SCALAR_VALUE).reshape(1, 1);


    @Param({"PLUS", "MINUS", "TIMES", "DIVIDE"})
    public Operation operation;

    @Param({"100", "10000", "1000000"})
    public double tensorLength;

    @Setup
    public void createTensor() {
        tensor = DoubleTensor.arange(0., tensorLength);
    }

    @Benchmark
    public long baseline() {
        DoubleTensor product = operation.apply(RANK_0_SCALAR_TENSOR, tensor);
        return product.getLength();
    }

    @Benchmark
    public long rank1() {
        DoubleTensor product = operation.apply(RANK_1_SCALAR_TENSOR, tensor);
        return product.getLength();
    }

    @Benchmark
    public long rank2() {
        DoubleTensor product = operation.apply(RANK_2_SCALAR_TENSOR, tensor);
        return product.getLength();
    }
}