package io.improbable.keanu.algorithms.mcmc;

import com.google.common.base.Preconditions;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.util.ProgressBar;
import io.improbable.keanu.vertices.VertexId;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import java.util.stream.Stream;

public class NetworkSamplesGenerator {

    private final SamplingAlgorithm algorithm;

    private int dropCount = 0;
    private int downSampleInterval = 1;

    private Supplier<ProgressBar> progressBarSupplier;

    public NetworkSamplesGenerator(SamplingAlgorithm algorithm, Supplier<ProgressBar> progressBarSupplier) {
        this.algorithm = algorithm;
        this.progressBarSupplier = progressBarSupplier;
    }

    public int getDropCount() {
        return dropCount;
    }

    /**
     * @param dropCount the number of samples to drop before collecting anything. If this is zero
     *                  then no samples will be dropped before collecting.
     * @return this NetworkSamplesGenerator set to drop the specified number of samples
     */
    public NetworkSamplesGenerator dropCount(int dropCount) {
        Preconditions.checkArgument(dropCount >= 0,
            "Drop count of %s is invalid. Cannot drop negative samples.",
            dropCount
        );
        this.dropCount = dropCount;
        return this;
    }

    public int getDownSampleInterval() {
        return downSampleInterval;
    }

    /**
     *
     * @param downSampleInterval collect 1 sample for every downSampleInterval.
     * @return
     */
    public NetworkSamplesGenerator downSampleInterval(int downSampleInterval) {
        Preconditions.checkArgument(downSampleInterval > 0,
            "Down-sample interval of %s is invalid. The down-sample interval means take every Nth sample." +
                " A down-sample interval of 1 would be no down-sampling.",
            downSampleInterval
        );

        this.downSampleInterval = downSampleInterval;
        return this;
    }

    public NetworkSamples generate(final int totalSampleCount) {
        Preconditions.checkArgument(dropCount < totalSampleCount,
            "Cannot drop more samples than requested or all of the samples. Samples requested %s and dropping %s",
            totalSampleCount, dropCount
        );

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

        }).onClose(progressBar::finish);
    }

    private void dropSamples(int dropCount, ProgressBar progressBar) {
        for (int i = 0; i < dropCount; i++) {
            algorithm.step();
            progressBar.progress("Dropping samples...", (i + 1) / (double) dropCount);
        }
    }

}
