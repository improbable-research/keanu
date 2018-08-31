package io.improbable.keanu.util.csv;

import static io.improbable.keanu.util.csv.WriteCsv.findLongestTensor;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;

public class RowWriter extends Writer {

    private static final String HEADER_STYLE = "[%s]";

    private List<? extends Vertex<? extends Tensor>> vertices;

    public RowWriter(List<? extends Vertex<? extends Tensor>> vertices, String emptyValue) {
        this.vertices = vertices;
        withEmptyValue(emptyValue);
    }

    public RowWriter(List<? extends Vertex<? extends Tensor>> vertices) {
        this(vertices, DEFAULT_EMPTY_VALUE);
    }

    @Override
    public File toFile(File file) {
        List<String[]> data = new ArrayList<>();
        int maxSize = findLongestTensor(vertices);

        for (Vertex<? extends Tensor> vertex : vertices) {
            List<String> row = new ArrayList<>();
            List<Object> flatList = vertex.getValue().asFlatList();
            for (int i = 0; i < maxSize; i++) {
                row.add(i < flatList.size() ? flatList.get(i).toString() : getEmptyValue());
            }
            String[] rowToString = new String[row.size()];
            data.add(row.toArray(rowToString));
        }
        return writeToFile(file, data);
    }

    @Override
    public Writer withDefaultHeader() {
        int sizeOfHeader = findLongestTensor(vertices);
        String[] header = createHeader(sizeOfHeader, HEADER_STYLE, i -> Integer.toString(i));
        withHeader(header);
        return this;
    }

}
