package io.improbable.keanu.benchmarks;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.tensor.dbl.ScalarDoubleTensor;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;

@State(Scope.Benchmark)
public class Nd4jDoubleTensorTimesScalar {

    public enum Operation {
        PLUS {
            public DoubleTensor apply(DoubleTensor lhs, DoubleTensor rhs) {
                return lhs.plus(rhs);
            }
        },
        MINUS {
            public DoubleTensor apply(DoubleTensor lhs, DoubleTensor rhs) {
                return lhs.minus(rhs);
            }
        },
        TIMES {
            public DoubleTensor apply(DoubleTensor lhs, DoubleTensor rhs) {
                return lhs.times(rhs);
            }
        },
        DIVIDE {
            public DoubleTensor apply(DoubleTensor lhs, DoubleTensor rhs) {
                return lhs.div(rhs);
            }
        };

        public abstract DoubleTensor apply(DoubleTensor lhs, DoubleTensor rhs);
    }

    @Param({"PLUS", "MINUS", "TIMES", "DIVIDE"})
    public Operation operation;

    @Param({"100", "10000", "1000000"})
    public double tensorLength;

    @Benchmark
    public long baseline() {
        DoubleTensor scalar = Nd4jDoubleTensor.scalar(1.);
        DoubleTensor tensor = DoubleTensor.arange(0., tensorLength);
        DoubleTensor product = operation.apply(scalar, tensor);
        return product.getLength();
    }

    @Benchmark
    public long rank0() {
        DoubleTensor scalar = Nd4jDoubleTensor.scalar(1.).reshape();
        DoubleTensor tensor = DoubleTensor.arange(0., tensorLength);
        DoubleTensor product = operation.apply(scalar, tensor);
        return product.getLength();
    }

    @Benchmark
    public long rank1() {
        DoubleTensor scalar = Nd4jDoubleTensor.scalar(1.).reshape(1);
        DoubleTensor tensor = DoubleTensor.arange(0., tensorLength);
        DoubleTensor product = operation.apply(scalar, tensor);
        return product.getLength();
    }

    @Benchmark
    public long rank2() {
        DoubleTensor scalar = Nd4jDoubleTensor.scalar(1.).reshape(1, 1);
        DoubleTensor tensor = DoubleTensor.arange(0., tensorLength);
        DoubleTensor product = operation.apply(scalar, tensor);
        return product.getLength();
    }
    @Benchmark
    public long customScalars() {
        DoubleTensor scalar = new ScalarDoubleTensor(1.);
        DoubleTensor tensor = DoubleTensor.arange(0., tensorLength);
        DoubleTensor product = operation.apply(scalar, tensor);
        return product.getLength();
    }
}