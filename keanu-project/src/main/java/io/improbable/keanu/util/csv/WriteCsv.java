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

    private WriteCsv() {
    }

    /**
     * @param samples the samples to be written to CSV
     * @param vertices the vertices whose samples will be written to CSV
     * @param <T> the type of the vertices
     * @return a writer for the csv file
     */
    public static <T extends Tensor> CsvWriter asSamples(NetworkSamples samples, List<Vertex<T>> vertices) {
        List<List<String>> data = new ArrayList<>();
        List<String> header = new ArrayList<>();
        String headerStyle = "%d" + "[%d]";
        boolean populateHeader = true;

        for (int i = 0; i < samples.size(); i++) {
            List<String> row = new ArrayList<>();
            for (Vertex<T> vertex : vertices) {
                T sample = samples.get(vertex).asList().get(i);
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
     * @param tensors the vertices whose values will be written to CSV in rows
     * @param <T> the type of the vertices
     * @return a writer for the csv file
     */
    public static <T extends Tensor> CsvWriter asRows(List<Vertex<T>> tensors) {
        List<List<String>> data = new ArrayList<>();
        List<String> header = new ArrayList<>();
        String headerStyle = "[%d]";
        int longestTensor = findLongestTensor(tensors);
        boolean populateHeader = true;

        for (Vertex<T> tensor : tensors) {
            List<String> row = new ArrayList<>();
            List<Object> flatList = tensor.getValue().asFlatList();
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
     * @param <T> the type of the vertices
     * @return a writer for the csv file
     */
    public static <T extends Tensor> CsvWriter asColumns(List<Vertex<T>> tensors) {
        List<List<String>> data = new ArrayList<>();
        List<String> header = new ArrayList<>();
        int longestTensor = findLongestTensor(tensors);
        boolean populateHeader = true;

        for (int i = 0; i < longestTensor; i++) {
            List<String> row = new ArrayList<>();
            for (Vertex<T> tensor : tensors) {
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

    private static <T extends Tensor> int findLongestTensor(List<Vertex<T>> tensors) {
        int longestTensor = 0;
        for (Vertex<T> tensor : tensors) {
            if (tensor.getValue().asFlatList().size() > longestTensor) {
                longestTensor = tensor.getValue().asFlatList().size();
            }
        }
        return longestTensor;
    }

}
