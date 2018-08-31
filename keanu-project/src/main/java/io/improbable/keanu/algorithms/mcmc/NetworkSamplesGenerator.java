package io.improbable.keanu.algorithms.mcmc;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.util.ProgressBar;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;

@Accessors(fluent = true)
public class NetworkSamplesGenerator {

    private final SamplingAlgorithm algorithm;

    @Getter
    @Setter
    private int dropCount = 0;

    @Getter
    @Setter
    private int downSampleInterval = 1;

    private ProgressBar progressBar = new ProgressBar();

    public NetworkSamplesGenerator(SamplingAlgorithm algorithm) {
        this.algorithm = algorithm;
    }

    public NetworkSamples generate(final int totalSampleCount) {

        Map<Long, List<?>> samplesByVertex = new HashMap<>();

        dropSamples(dropCount);

        int sampleCount = 0;
        int samplesLeft = totalSampleCount - dropCount;
        for (int i = 0; i < samplesLeft; i++) {
            if (i % downSampleInterval == 0) {
                algorithm.sample(samplesByVertex);
                sampleCount++;
            } else {
                algorithm.step();
            }

            progressBar.progress("Sampling...", (i + 1) / (double) samplesLeft);
        }

        progressBar.finish();
        return new NetworkSamples(samplesByVertex, sampleCount);
    }

    public Stream<NetworkState> stream() {

        dropSamples(dropCount);

        final AtomicInteger sampleNumber = new AtomicInteger(0);

        return Stream.generate(() -> {

            sampleNumber.getAndIncrement();

            for (int i = 0; i < downSampleInterval - 1; i++) {
                algorithm.step();
            }

            NetworkState sample = algorithm.sample();
            progressBar.progress(String.format("Sample #%,d completed", sampleNumber.get()));
            return sample;

        }).onClose(() -> progressBar.finish());
    }

    private void dropSamples(int dropCount) {
        for (int i = 0; i < dropCount; i++) {
            algorithm.step();
            progressBar.progress("Dropping samples...", (i + 1) / (double) dropCount);
        }
    }

}
