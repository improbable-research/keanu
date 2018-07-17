package io.improbable.keanu.util.csv;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class SampleWriter extends Writer {

    private static final String HEADER_STYLE = "{%d}[%d]";

    private NetworkSamples samples;
    private List<? extends Vertex<? extends Tensor>> vertices;

    public SampleWriter(NetworkSamples samples, List<? extends Vertex<? extends Tensor>> vertices) {
        this.samples = samples;
        this.vertices = vertices;
    }

    @Override
    File toFile(String file) {
        List<String[]> data = new ArrayList<>();

        for (int i = 0; i < samples.size(); i++) {
            List<String> row = new ArrayList<>();
            for (Vertex<? extends Tensor> vertex : vertices) {
                Tensor sample = samples.get(vertex).asList().get(i);
                List<Object> flatList = sample.asFlatList();
                for (int j = 0; j < flatList.size(); j++) {
                    row.add(flatList.get(j).toString());
                }
            }
            String[] rowToString = new String[row.size()];
            data.add(row.toArray(rowToString));
        }

        return writeToFile(file, data);
    }

    @Override
    Writer withDefaultHeader() {
        List<String> header = new ArrayList<>();
        for (Vertex<? extends Tensor> vertex : vertices) {
            for (int j = 0; j < vertex.getValue().getLength(); j++) {
                header.add(String.format(HEADER_STYLE, vertex.getId(), j));
            }
        }
        String[] headerToArray = new String[header.size()];
        withHeader(header.toArray(headerToArray));
        return this;
    }
}
