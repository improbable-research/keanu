package io.improbable.keanu.algorithms.mcmc;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Mode;

public class NUTSPerfTest {

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }

    @Fork(value = 1, warmups = 1)
    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void testHMC() {
        HamiltonianTest test = new HamiltonianTest();
        test.setup();
        test.samplesGaussian();
    }

    @Fork(value = 1, warmups = 1)
    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void testNUTS() {
        EfficientNUTSTest test = new EfficientNUTSTest();
        test.setup();
        test.samplesGaussian();
    }

}
