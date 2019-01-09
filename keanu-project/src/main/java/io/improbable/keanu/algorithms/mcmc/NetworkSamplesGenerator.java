package io.improbable.keanu.algorithms.mcmc;

import com.google.common.base.Preconditions;
import io.improbable.keanu.algorithms.NetworkSample;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.util.status.ProgressStatusBar;
import io.improbable.keanu.util.status.StatusBar;
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

    private Supplier<StatusBar> statusBarSupplier;

    public NetworkSamplesGenerator(SamplingAlgorithm algorithm, Supplier<StatusBar> statusBarSupplier) {
        this.algorithm = algorithm;
        this.statusBarSupplier = statusBarSupplier;
    }

    public int getDropCount() {
        return dropCount;
    }

    /**
     * @param dropCount the number of samples to drop before collecting anything. If this is zero
     *                  then no samples will be dropped before collecting.
     * @return this NetworkSamplesGenerator set to drop the specified number of samples
     * @throws IllegalArgumentException when dropCount is less than zero
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
     * @param downSampleInterval collect 1 sample for every downSampleInterval. If this is 1 then there will be no
     *                           down-sampling. If this is 2 then every other sample will be taken. If this is 3 then
     *                           2 samples will be dropped before one is taken.
     * @return this NetworkSamplesGenerator set to down-sample at the specified downSampleInterval
     * @throws IllegalArgumentException when downSampleInterval is less than or equal to zero
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

    /**
     * @param totalSampleCount The total number of samples to generate. This is the total before any dropping
     *                         or down-sampling is done. If you drop 10 and down sample 2 and request a totalSampleCount
     *                         of 100 then you would take 100 samples, drop 10 and then take every other sample resulting
     *                         in 45 samples returned.
     * @return Samples after dropping and down-sampling.
     */
    public NetworkSamples generate(final int totalSampleCount) {
        Preconditions.checkArgument(dropCount < totalSampleCount,
            "Cannot drop more samples than requested or all of the samples. Samples requested %s and dropping %s",
            totalSampleCount, dropCount
        );

        StatusBar statusBar = statusBarSupplier.get();

        Map<VertexId, List<?>> samplesByVertex = new HashMap<>();
        List<Double> logOfMasterPForEachSample = new ArrayList<>();

        dropSamples(dropCount, statusBar);

        ProgressStatusBar progressBar = new ProgressStatusBar(statusBar);
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

        statusBar.finish();
        return new NetworkSamples(samplesByVertex, logOfMasterPForEachSample, sampleCount);
    }

    /**
     * @return A stream of samples starting after dropping. Down-sampling is handled outside of the stream (i.e. the
     * stream will be the final result after dropping and down-sampling)
     */
    public Stream<NetworkSample> stream() {

        StatusBar statusBar = statusBarSupplier.get();

        dropSamples(dropCount, statusBar);

        final AtomicInteger sampleNumber = new AtomicInteger(0);
        return Stream.generate(() -> {

            sampleNumber.getAndIncrement();

            for (int i = 0; i < downSampleInterval - 1; i++) {
                algorithm.step();
            }

            NetworkSample sample = algorithm.sample();
            statusBar.setMessage(String.format("Sample #%,d completed", sampleNumber.get()));
            return sample;

        }).onClose(statusBar::finish);
    }

    private void dropSamples(int dropCount, StatusBar statusBar) {
        ProgressStatusBar progressBar = new ProgressStatusBar(statusBar);
        for (int i = 0; i < dropCount; i++) {
            algorithm.step();
            progressBar.progress("Dropping samples...", (i + 1) / (double) dropCount);
        }
    }

}
