package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSample;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.util.status.StatusBar;
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
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;

public class NetworkSamplesGeneratorTest {

    @Test
    public void dropsAndSamplesExpectedNumberOfStepsOnGeneration() {

        AtomicInteger stepCount = new AtomicInteger(0);
        AtomicInteger sampleCount = new AtomicInteger(0);

        TestSamplingAlgorithm algorithm = new TestSamplingAlgorithm(stepCount, sampleCount);
        NetworkSamplesGenerator unitUnderTest = new NetworkSamplesGenerator(algorithm, StatusBar::new);

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
        NetworkSamplesGenerator unitUnderTest = new NetworkSamplesGenerator(algorithm, StatusBar::new);

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
    public void doesUpdateStatusAndFinishStatusOnGeneration() {
        AtomicInteger stepCount = new AtomicInteger(0);
        AtomicInteger sampleCount = new AtomicInteger(0);

        StatusBar statusBar = mock(StatusBar.class);
        TestSamplingAlgorithm algorithm = new TestSamplingAlgorithm(stepCount, sampleCount);

        NetworkSamplesGenerator unitUnderTest = new NetworkSamplesGenerator(algorithm, () -> statusBar);
        unitUnderTest.generate(10);

        Mockito.verify(statusBar, times(1)).setMessage(anyString());
        Mockito.verify(statusBar).finish();
    }


    @Test
    public void doesCreateNewStatusBarOnGenerationFinish() {
        AtomicInteger stepCount = new AtomicInteger(0);
        AtomicInteger sampleCount = new AtomicInteger(0);

        StatusBar statusBar1 = mock(StatusBar.class);
        StatusBar statusBar2 = mock(StatusBar.class);

        AtomicInteger statusBarCreationCount = new AtomicInteger(0);
        TestSamplingAlgorithm algorithm = new TestSamplingAlgorithm(stepCount, sampleCount);

        NetworkSamplesGenerator unitUnderTest = new NetworkSamplesGenerator(algorithm, () -> {
            int callNumber = statusBarCreationCount.getAndIncrement();
            if (callNumber == 0) {
                return statusBar1;
            } else {
                return statusBar2;
            }
        });

        unitUnderTest.generate(10);
        Mockito.verify(statusBar1, times(1)).setMessage(anyString());
        Mockito.verify(statusBar1).finish();

        unitUnderTest.generate(8);
        Mockito.verify(statusBar2, times(1)).setMessage(anyString());
        Mockito.verify(statusBar2).finish();
    }

    @Test
    public void doesUpdateProgressAndFinishProgressWhenStreaming() {
        StatusBar progressBar = mock(StatusBar.class);
        TestSamplingAlgorithm algorithm = new TestSamplingAlgorithm(new AtomicInteger(0), new AtomicInteger(0));
        Stream<NetworkSample> sampleStream = new NetworkSamplesGenerator(algorithm, () -> progressBar).stream();
        sampleStream.limit(10).count();
        sampleStream.close();

        Mockito.verify(progressBar, times(10)).setMessage(anyString());
        Mockito.verify(progressBar).finish();
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesNotAllowZeroDownSample() {
        TestSamplingAlgorithm algorithm = new TestSamplingAlgorithm(new AtomicInteger(0), new AtomicInteger(0));
        NetworkSamplesGenerator unitUnderTest = new NetworkSamplesGenerator(algorithm, StatusBar::new);
        unitUnderTest.downSampleInterval(0).stream();
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesNotAllowNegativeDropCount() {
        TestSamplingAlgorithm algorithm = new TestSamplingAlgorithm(new AtomicInteger(0), new AtomicInteger(0));
        NetworkSamplesGenerator unitUnderTest = new NetworkSamplesGenerator(algorithm, StatusBar::new);
        unitUnderTest.dropCount(-10).generate(100);
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesNotAllowDroppingMoreThanRequesting() {
        TestSamplingAlgorithm algorithm = new TestSamplingAlgorithm(new AtomicInteger(0), new AtomicInteger(0));
        NetworkSamplesGenerator unitUnderTest = new NetworkSamplesGenerator(algorithm, StatusBar::new);
        unitUnderTest.dropCount(200).generate(100);
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
        public NetworkSample sample() {
            sampleCount.incrementAndGet();
            return new NetworkSample(Collections.emptyMap(), 0.0);
        }
    }
}
