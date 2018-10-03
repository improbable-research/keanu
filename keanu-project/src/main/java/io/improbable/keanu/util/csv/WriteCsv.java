package io.improbable.keanu.util.csv;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import java.util.List;

/** This class provides static helper functions for writing csv data */
public class WriteCsv {

    /**
     * @param samples the samples to be written to CSV
     * @param vertices the vertices whose samples will be written to CSV
     * @return a writer for the csv file
     */
    public static SampleWriter asSamples(
            NetworkSamples samples, List<? extends Vertex<? extends Tensor>> vertices) {
        return new SampleWriter(samples, vertices);
    }

    /**
     * @param vertices the vertices whose values will be written to CSV in rows
     * @return a writer for the csv file
     */
    public static RowWriter asRows(List<? extends Vertex<? extends Tensor>> vertices) {
        return new RowWriter(vertices);
    }

    /**
     * @param vertices the vertices whose values will be written to CSV in columns
     * @return a writer for the csv file
     */
    public static ColumnWriter asColumns(List<? extends Vertex<? extends Tensor>> vertices) {
        return new ColumnWriter(vertices);
    }

    public static int findLongestTensor(List<? extends Vertex<? extends Tensor>> tensors) {
        int longestTensor = 0;
        for (Vertex<? extends Tensor> tensor : tensors) {
            if (tensor.getValue().getLength() > longestTensor) {
                longestTensor = tensor.getValue().asFlatList().size();
            }
        }
        return longestTensor;
    }
}
