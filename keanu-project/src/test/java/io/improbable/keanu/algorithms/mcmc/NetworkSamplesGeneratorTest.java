package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.network.SimpleNetworkState;
import io.improbable.keanu.util.ProgressBar;
import io.improbable.keanu.vertices.VertexId;
import lombok.Value;
import org.junit.Test;
import org.mockito.Mockito;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;

public class NetworkSamplesGeneratorTest {

    @Test
    public void dropsAndSamplesExpectedNumberOfStepsOnGeneration() {

        AtomicInteger stepCount = new AtomicInteger(0);
        AtomicInteger sampleCount = new AtomicInteger(0);

        TestSamplingAlgorithm algorithm = new TestSamplingAlgorithm(stepCount, sampleCount);
        NetworkSamplesGenerator unitUnderTest = new NetworkSamplesGenerator(algorithm, ProgressBar::new);

        int totalGenerated = 12;
        int dropCount = 3;
        int downSampleInterval = 2;
        unitUnderTest.dropCount(dropCount).downSampleInterval(downSampleInterval);
        NetworkSamples samples = unitUnderTest.generate(totalGenerated);

        int expectedCollected = (int) Math.ceil((totalGenerated - dropCount) / (double) downSampleInterval);
        assertEquals(totalGenerated, algorithm.stepCount.get() + algorithm.sampleCount.get());
        assertEquals(expectedCollected, samples.size());
    }

    @Test
    public void streamsExpectedNumberOfSamples() {

        AtomicInteger stepCount = new AtomicInteger(0);
        AtomicInteger sampleCount = new AtomicInteger(0);

        TestSamplingAlgorithm algorithm = new TestSamplingAlgorithm(stepCount, sampleCount);
        NetworkSamplesGenerator unitUnderTest = new NetworkSamplesGenerator(algorithm, ProgressBar::new);

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

    @Test
    public void doesUpdateProgressAndFinishProgressOnGeneration() {
        AtomicInteger stepCount = new AtomicInteger(0);
        AtomicInteger sampleCount = new AtomicInteger(0);

        ProgressBar progressBar = mock(ProgressBar.class);
        TestSamplingAlgorithm algorithm = new TestSamplingAlgorithm(stepCount, sampleCount);

        NetworkSamplesGenerator unitUnderTest = new NetworkSamplesGenerator(algorithm, () -> progressBar);
        unitUnderTest.generate(10);

        Mockito.verify(progressBar, times(10)).progress(anyString(), anyDouble());
        Mockito.verify(progressBar).finish();
    }

    @Test
    public void doesCreateNewProgressBarOnGenerationFinish() {
        AtomicInteger stepCount = new AtomicInteger(0);
        AtomicInteger sampleCount = new AtomicInteger(0);

        ProgressBar progressBar1 = mock(ProgressBar.class);
        ProgressBar progressBar2 = mock(ProgressBar.class);

        AtomicInteger progressBarCreationCount = new AtomicInteger(0);
        TestSamplingAlgorithm algorithm = new TestSamplingAlgorithm(stepCount, sampleCount);

        NetworkSamplesGenerator unitUnderTest = new NetworkSamplesGenerator(algorithm, () -> {
            int callNumber = progressBarCreationCount.getAndIncrement();
            if (callNumber == 0) {
                return progressBar1;
            } else {
                return progressBar2;
            }
        });

        unitUnderTest.generate(10);
        Mockito.verify(progressBar1, times(10)).progress(anyString(), anyDouble());
        Mockito.verify(progressBar1).finish();

        unitUnderTest.generate(8);
        Mockito.verify(progressBar2, times(8)).progress(anyString(), anyDouble());
        Mockito.verify(progressBar2).finish();
    }

    @Test
    public void doesUpdateProgressAndFinishProgressWhenStreaming() {
        ProgressBar progressBar = mock(ProgressBar.class);
        TestSamplingAlgorithm algorithm = new TestSamplingAlgorithm(new AtomicInteger(0), new AtomicInteger(0));
        Stream<NetworkState> sampleStream = new NetworkSamplesGenerator(algorithm, () -> progressBar).stream();
        sampleStream.limit(10).count();
        sampleStream.close();

        Mockito.verify(progressBar, times(10)).progress(anyString());
        Mockito.verify(progressBar).finish();
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
        public void sample(Map<VertexId, List<?>> samples, List<Double> logOfMasterPForEachSample) {
            sampleCount.incrementAndGet();
        }

        @Override
        public NetworkState sample() {
            sampleCount.incrementAndGet();
            return new SimpleNetworkState(Collections.emptyMap());
        }
    }
}
