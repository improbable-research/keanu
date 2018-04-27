package io.improbable.keanu.algorithms.mcmc;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Mode;

public class NUTSPerformanceTest {

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }

    @Fork(value = 1, warmups = 1)
    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void simpleGaussianSamplingHMCTest() {
        HamiltonianTest test = new HamiltonianTest();
        test.setup();
        test.samplesGaussian();
    }

    @Fork(value = 1, warmups = 1)
    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void simpleGaussianSamplingNUTSTest() {
        NUTSTest test = new NUTSTest();
        test.setup();
        test.samplesGaussian();
    }

}
