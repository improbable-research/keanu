package io.improbable.keanu.benchmarks;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.tensor.dbl.ScalarDoubleTensor;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;

@State(Scope.Benchmark)
public class Nd4jDoubleTensorScalarOperations {

    public DoubleTensor tensor;

    public static DoubleTensor[] scalars = new DoubleTensor[] {
        Nd4jDoubleTensor.scalar(1.),
        Nd4jDoubleTensor.scalar(1.).reshape(1),
        Nd4jDoubleTensor.scalar(1.).reshape(1, 1),
        new ScalarDoubleTensor(1.)
    };



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
        DoubleTensor product = operation.apply(scalars[0], tensor);
        return product.getLength();
    }

    @Benchmark
    public long rank1() {
        DoubleTensor product = operation.apply(scalars[1], tensor);
        return product.getLength();
    }

    @Benchmark
    public long rank2() {
        DoubleTensor product = operation.apply(scalars[2], tensor);
        return product.getLength();
    }
    @Benchmark
    public long customScalars() {
        DoubleTensor product = operation.apply(scalars[3], tensor);
        return product.getLength();
    }

}