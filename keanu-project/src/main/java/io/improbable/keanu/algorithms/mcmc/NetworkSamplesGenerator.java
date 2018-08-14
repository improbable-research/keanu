package io.improbable.keanu.algorithms.mcmc;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.NetworkState;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;

@Accessors(fluent = true)
public class NetworkSamplesGenerator {

    private final int totalCount;
    private final SamplingAlgorithm algorithm;

    @Getter
    @Setter
    private int dropCount;

    @Getter
    @Setter
    private int downSampleInterval;


    public NetworkSamplesGenerator(int totalCount, SamplingAlgorithm algorithm) {
        this.totalCount = totalCount;
        this.algorithm = algorithm;
    }

    public NetworkSamples generate() {

        Map<Long, List<?>> samplesByVertex = new HashMap<>();

        dropSamples(dropCount);

        int sampleCount = 0;
        int samplesLeft = totalCount - dropCount;
        for (int i = 0; i < samplesLeft; i++) {
            if (i % downSampleInterval == 0) {
                algorithm.sample(samplesByVertex);
                sampleCount++;
            } else {
                algorithm.step();
            }
        }

        return new NetworkSamples(samplesByVertex, sampleCount);
    }

    public Stream<NetworkState> stream() {

        dropSamples(dropCount);

        return Stream.generate(() -> {

            for (int i = 0; i < downSampleInterval; i++) {
                algorithm.step();
            }

            return algorithm.sample();
        });
    }

    private void dropSamples(int dropCount) {
        for (int i = 0; i < dropCount; i++) {
            algorithm.step();
        }
    }

    public interface SamplingAlgorithm {

        void step();

        void sample(Map<Long, List<?>> samples);

        NetworkState sample();
    }
}
