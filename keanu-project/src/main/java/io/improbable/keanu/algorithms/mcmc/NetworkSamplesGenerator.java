package io.improbable.keanu.algorithms.mcmc;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
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

            progressBar.progress();
        }

        progressBar.finished();
        return new NetworkSamples(samplesByVertex, sampleCount);
    }

    public Stream<NetworkState> stream() {

        dropSamples(dropCount);

        return Stream.generate(() -> {

            for (int i = 0; i < downSampleInterval - 1; i++) {
                algorithm.step();
                progressBar.progress();
            }

            progressBar.progress();
            return algorithm.sample();

        }).onClose(() -> progressBar.finished());
    }

    private void dropSamples(int dropCount) {
        for (int i = 0; i < dropCount; i++) {
            algorithm.step();
            progressBar.progress();
        }
    }

}
