package io.improbable.keanu.algorithms.mcmc;

import static org.junit.Assert.assertEquals;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import org.junit.Test;

import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.network.SimpleNetworkState;
import lombok.Value;

public class NetworkSamplesGeneratorTest {

    @Test
    public void dropsAndSamplesExpectedNumberOfStepsOnGeneration() {

        AtomicInteger stepCount = new AtomicInteger(0);
        AtomicInteger sampleCount = new AtomicInteger(0);

        TestSamplingAlgorithm algorithm = new TestSamplingAlgorithm(stepCount, sampleCount);
        NetworkSamplesGenerator unitUnderTest = new NetworkSamplesGenerator(algorithm);

        int totalGenerated = 12;
        int dropCount = 3;
        int downSampleInterval = 2;
        unitUnderTest.dropCount(dropCount).downSampleInterval(downSampleInterval);
        unitUnderTest.generate(totalGenerated);

        int expectedCollected = (int) Math.ceil((totalGenerated - dropCount) / (double) downSampleInterval);
        assertEquals(totalGenerated, algorithm.stepCount.get() + algorithm.sampleCount.get());
        assertEquals(expectedCollected, algorithm.sampleCount.get());
    }

    @Test
    public void streamsExpectedNumberOfSamples() {

        AtomicInteger stepCount = new AtomicInteger(0);
        AtomicInteger sampleCount = new AtomicInteger(0);

        TestSamplingAlgorithm algorithm = new TestSamplingAlgorithm(stepCount, sampleCount);
        NetworkSamplesGenerator unitUnderTest = new NetworkSamplesGenerator(algorithm);

        int totalCollected = 5;
        int dropCount = 3;
        int downSampleInterval = 2;
        unitUnderTest.dropCount(dropCount).downSampleInterval(downSampleInterval);
        unitUnderTest.stream()
            .limit(totalCollected)
            .collect(Collectors.toList());

        //expected step + sample count differs from generate case due to different behaviour
        int expectedTotal = dropCount + totalCollected * downSampleInterval;
        assertEquals(expectedTotal, algorithm.stepCount.get() + algorithm.sampleCount.get());
        assertEquals(totalCollected, algorithm.sampleCount.get());
    }

    @Value
    public static class TestSamplingAlgorithm implements SamplingAlgorithm {

        private final AtomicInteger stepCount;
        private final AtomicInteger sampleCount;

        @Override
        public void step() {
            stepCount.incrementAndGet();
        }

        @Override
        public void sample(Map<Long, List<?>> samples) {
            sampleCount.incrementAndGet();
        }

        @Override
        public NetworkState sample() {
            sampleCount.incrementAndGet();
            return new SimpleNetworkState(Collections.emptyMap());
        }
    }
}
