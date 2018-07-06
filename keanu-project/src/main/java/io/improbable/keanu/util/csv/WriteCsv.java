package io.improbable.keanu.util.csv;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.ArrayList;
import java.util.List;

/**
 * This class provides static helper functions for writing csv data
 */
public class WriteCsv {

    private static String DEFAULT_EMPTY_VALUE = "-";

    /**
     * @param samples the samples to be written to CSV
     * @param vertices the vertices whose samples will be written to CSV
     * @return a writer for the csv file
     */
    public static CsvWriter asSamples(NetworkSamples samples, List<? extends Vertex<? extends Tensor>> vertices) {
        List<List<String>> data = new ArrayList<>();
        List<String> header = new ArrayList<>();
        String headerStyle = "%d" + "[%d]";
        boolean populateHeader = true;

        for (int i = 0; i < samples.size(); i++) {
            List<String> row = new ArrayList<>();
            for (Vertex<? extends Tensor> vertex : vertices) {
                Tensor sample = samples.get(vertex).asList().get(i);
                List<Object> flatList = sample.asFlatList();
                for (int j = 0; j < flatList.size(); j++) {
                    row.add(flatList.get(j).toString());
                    if (populateHeader) {
                        header.add(String.format(headerStyle, vertex.getId(), j));
                    }
                }
            }
            populateHeader = false;
            data.add(row);
        }

        return new CsvWriter(data).withCustomHeader(header).disableHeader();
    }

    /**
     * @param vertices the vertices whose values will be written to CSV in rows
     * @return a writer for the csv file
     */
    public static CsvWriter asRows(List<? extends Vertex<? extends Tensor>> vertices) {
        List<List<String>> data = new ArrayList<>();
        List<String> header = new ArrayList<>();
        String headerStyle = "[%d]";
        int longestTensor = findLongestTensor(vertices);
        boolean populateHeader = true;

        for (Vertex<? extends Tensor> vertex : vertices) {
            List<String> row = new ArrayList<>();
            List<Object> flatList = vertex.getValue().asFlatList();
            for (int i = 0; i < longestTensor; i++) {
                if (populateHeader) {
                    header.add(String.format(headerStyle, i));
                }
                if (i < flatList.size()) {
                    row.add(flatList.get(i).toString());
                } else {
                    row.add(DEFAULT_EMPTY_VALUE);
                }
            }
            populateHeader = false;
            data.add(row);
        }
        return new CsvWriter(data).withCustomHeader(header).disableHeader();
    }

    /**
     * @param tensors the vertices whose values will be written to CSV in columns
     * @return a writer for the csv file
     */
    public static CsvWriter asColumns(List<? extends Vertex<? extends Tensor>> tensors) {
        List<List<String>> data = new ArrayList<>();
        List<String> header = new ArrayList<>();
        int longestTensor = findLongestTensor(tensors);
        boolean populateHeader = true;

        for (int i = 0; i < longestTensor; i++) {
            List<String> row = new ArrayList<>();
            for (Vertex<? extends Tensor> tensor : tensors) {
                List<Object> flatList = tensor.getValue().asFlatList();
                if (populateHeader) {
                    header.add(String.valueOf(tensor.getId()));
                }
                if (i < flatList.size()) {
                    row.add(flatList.get(i).toString());
                } else {
                    row.add(DEFAULT_EMPTY_VALUE);
                }

            }
            populateHeader = false;
            data.add(row);
        }
        return new CsvWriter(data).withCustomHeader(header).disableHeader();

    }

    private static int findLongestTensor(List<? extends Vertex<? extends Tensor>> tensors) {
        int longestTensor = 0;
        for (Vertex<? extends Tensor> tensor : tensors) {
            if (tensor.getValue().getLength() > longestTensor) {
                longestTensor = tensor.getValue().asFlatList().size();
            }
        }
        return longestTensor;
    }

}
