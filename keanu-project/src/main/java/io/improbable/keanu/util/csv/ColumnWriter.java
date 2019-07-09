package io.improbable.keanu.util.csv;

import com.opencsv.CSVWriter;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static io.improbable.keanu.util.csv.WriteCsv.findLongestTensor;

public class ColumnWriter extends Writer {

    private static final String HEADER_STYLE = "{%s}";

    private List<? extends Vertex<? extends Tensor>> vertices;

    public ColumnWriter(List<? extends Vertex<? extends Tensor>> vertices, String emptyValue) {
        this.vertices = vertices;
        withEmptyValue(emptyValue);
    }

    public ColumnWriter(List<? extends Vertex<? extends Tensor>> vertices) {
        this(vertices, DEFAULT_EMPTY_VALUE);
    }

    @Override
    public File toFile(File file) throws IOException {
        try (CSVWriter writer = prepareWriter(file)) {
            int maxSize = findLongestTensor(vertices);

            for (int i = 0; i < maxSize; i++) {
                List<String> row = new ArrayList<>();
                for (Vertex<? extends Tensor> vertex : vertices) {
                    List<Object> flatList = vertex.getValue().asFlatList();
                    row.add(i < flatList.size() ? flatList.get(i).toString() : getEmptyValue());
                }
                String[] rowArray = new String[row.size()];
                writer.writeNext(row.toArray(rowArray), false);
            }
        }
        return file;
    }

    @Override
    public Writer withDefaultHeader() {
        int headerSize = vertices.size();
        String[] header = createHeader(headerSize, HEADER_STYLE, i -> vertices.get(i).getId().toString());
        withHeader(header);
        return this;
    }

}
