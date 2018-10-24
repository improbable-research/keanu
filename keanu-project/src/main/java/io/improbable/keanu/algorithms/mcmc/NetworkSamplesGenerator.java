package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.util.ProgressBar;
import io.improbable.keanu.vertices.VertexId;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import java.util.stream.Stream;

@Accessors(fluent = true)
public class NetworkSamplesGenerator {

    private final SamplingAlgorithm algorithm;

    @Getter
    @Setter
    private int dropCount = 0;

    @Getter
    @Setter
    private int downSampleInterval = 1;

    private Supplier<ProgressBar> progressBarSupplier;

    public NetworkSamplesGenerator(SamplingAlgorithm algorithm, Supplier<ProgressBar> progressBarSupplier) {
        this.algorithm = algorithm;
        this.progressBarSupplier = progressBarSupplier;
    }

    public NetworkSamples generate(final int totalSampleCount) {

        ProgressBar progressBar = progressBarSupplier.get();

        Map<VertexId, List<?>> samplesByVertex = new HashMap<>();
        List<Double> logOfMasterPForEachSample = new ArrayList<>();

        dropSamples(dropCount, progressBar);

        int sampleCount = 0;
        int samplesLeft = totalSampleCount - dropCount;
        for (int i = 0; i < samplesLeft; i++) {
            if (i % downSampleInterval == 0) {
                algorithm.sample(samplesByVertex, logOfMasterPForEachSample);
                sampleCount++;
            } else {
                algorithm.step();
            }

            progressBar.progress("Sampling...", (i + 1) / (double) samplesLeft);
        }

        progressBar.finish();
        return new NetworkSamples(samplesByVertex, logOfMasterPForEachSample, sampleCount);
    }

    public Stream<NetworkState> stream() {

        ProgressBar progressBar = progressBarSupplier.get();

        dropSamples(dropCount, progressBar);

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

    private void dropSamples(int dropCount, ProgressBar progressBar) {
        for (int i = 0; i < dropCount; i++) {
            algorithm.step();
            progressBar.progress("Dropping samples...", (i + 1) / (double) dropCount);
        }
    }

}
