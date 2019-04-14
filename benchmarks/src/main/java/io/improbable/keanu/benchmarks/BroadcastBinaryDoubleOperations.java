package io.improbable.keanu.benchmarks;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;

import java.util.Arrays;

@State(Scope.Benchmark)
public class BroadcastBinaryDoubleOperations {

    private static final int NUM_OPERATIONS = 1000;

    @Param({"TIMES", "DIVIDE"})
    public BinaryOperation operation;

    @Param({"1x1,2x10", "2x10,1x1", "1x10,2x10", "2x10,1x10", "2x2x2,2x2", "2x2,2x2x2"})
    public String dims;

    @Param({"JVM", "ND4J"})
    public DoubleTensorImpl impl;

    DoubleTensor left;
    DoubleTensor right;

    @Setup
    public void setup() {

        DoubleTensor.setFactory(impl.getFactory());

        String[] dimTokens = dims.split(",");

        String leftParam = dimTokens[0];
        String rightParam = dimTokens[1];

        long[] leftShape = Arrays.stream(leftParam.split("x"))
            .mapToLong(Long::parseLong)
            .toArray();

        long[] rightShape = Arrays.stream(rightParam.split("x"))
            .mapToLong(Long::parseLong)
            .toArray();

        left = KeanuRandom.getDefaultRandom().nextGaussian(leftShape);
        right = KeanuRandom.getDefaultRandom().nextGaussian(rightShape);
    }

    @Benchmark
    public DoubleTensor benchmark() {

        DoubleTensor result = null;
        result = operation.apply(left, right);

        return result;
    }
}
