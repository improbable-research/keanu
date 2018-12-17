package io.improbable.keanu.tensor.dbl;

import org.openjdk.jmh.annotations.Benchmark;

public class Nd4jDoubleTensorTimesScalar {

    @Benchmark
    public long baseline() {
        DoubleTensor scalar = Nd4jDoubleTensor.scalar(1.);
        DoubleTensor tensor = DoubleTensor.arange(0., 1000.);
        DoubleTensor product = scalar.times(tensor);
        return product.getLength();
    }

    @Benchmark
    public long customScalars() {
        DoubleTensor scalar = new ScalarDoubleTensor(1.);
        DoubleTensor tensor = DoubleTensor.arange(0., 1000.);
        DoubleTensor product = scalar.times(tensor);
        return product.getLength();
    }
}