package io.improbable.keanu.algorithms.mcmc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.vertices.VertexId;
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


    public NetworkSamplesGenerator(SamplingAlgorithm algorithm) {
        this.algorithm = algorithm;
    }

    public NetworkSamples generate(final int totalSampleCount) {

        Map<VertexId, List<?>> samplesByVertex = new HashMap<>();
        List<Double> logProbs = new ArrayList<>();

        dropSamples(dropCount);

        int sampleCount = 0;
        int samplesLeft = totalSampleCount - dropCount;
        for (int i = 0; i < samplesLeft; i++) {
            if (i % downSampleInterval == 0) {
                algorithm.sample(samplesByVertex, logProbs);
                sampleCount++;
            } else {
                algorithm.step();
            }
        }

        return new NetworkSamples(samplesByVertex, logProbs, sampleCount);
    }

    public Stream<NetworkState> stream() {

        dropSamples(dropCount);

        return Stream.generate(() -> {

            for (int i = 0; i < downSampleInterval - 1; i++) {
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

}
