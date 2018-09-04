package io.improbable.keanu.util.csv;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;

public class WriteCsvTest {

    private NetworkSamples samples;
    private List<Vertex<DoubleTensor>> rowTensors = new ArrayList<>();
    private List<Vertex<DoubleTensor>> columnTensors = new ArrayList<>();
    private List<Vertex<DoubleTensor>> scalarTensors = new ArrayList<>();
    private List<Vertex<IntegerTensor>> integerColumnTensors = new ArrayList<>();

    @Before
    public void setup() {
        GaussianVertex g1 = new GaussianVertex(new int[]{1, 5}, 0, 1);
        GaussianVertex g2 = new GaussianVertex(new int[]{1, 4}, 0, 1);

        GaussianVertex f1 = new GaussianVertex(new int[]{5, 1}, 0, 1);
        GaussianVertex f2 = new GaussianVertex(new int[]{4, 1}, 0, 1);

        g1.setValue(new double[]{1, 2, 3, 4, 5});
        g2.setValue(new double[]{5, 4, 3, 2});

        f1.setValue(new double[]{1, 2, 3, 4, 5});
        f2.setValue(new double[]{5, 4, 3, 2});

        ConstantDoubleVertex c1 = new ConstantDoubleVertex(0.5);
        ConstantDoubleVertex c2 = new ConstantDoubleVertex(1.5);

        ConstantIntegerVertex i1 = new ConstantIntegerVertex(new int[]{1, 2, 3});
        ConstantIntegerVertex i2 = new ConstantIntegerVertex(new int[]{3, 2, 1});

        rowTensors.addAll(Arrays.asList(g1, g2));
        columnTensors.addAll(Arrays.asList(f1, f2));
        scalarTensors.addAll(Arrays.asList(c1, c2));
        integerColumnTensors.addAll(Arrays.asList(i1, i2));

        Map<VertexId, List<DoubleTensor>> networkSamples = new HashMap<>();
        networkSamples.put(g1.getId(), Arrays.asList(g1.getValue(), g1.times(2).getValue()));
        networkSamples.put(g2.getId(), Arrays.asList(g2.getValue(), g2.times(2).getValue()));
        samples = new NetworkSamples(networkSamples, new ArrayList<>(), 2);
    }

    @Test
    public void writeSamplesToCsv() throws IOException {
        File file = WriteCsv.asSamples(samples, rowTensors).toFile(File.createTempFile("test",".csv"));

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(false);
        List<List<String>> lines = reader.readLines();

        assertTrue(lines.size() == 2);
        assertTrue(lines.get(0).equals(
            Arrays.asList("1.0", "2.0", "3.0", "4.0", "5.0", "5.0", "4.0", "3.0", "2.0")));
        assertTrue(lines.get(1).equals(
            Arrays.asList("2.0", "4.0", "6.0", "8.0", "10.0", "10.0", "8.0", "6.0", "4.0")));

        file.delete();
    }

    @Test
    public void writeSamplesToCsvWithHeader() throws IOException {
        File file = WriteCsv.asSamples(samples, rowTensors).withDefaultHeader().toFile(File.createTempFile("test",".csv"));

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(true);
        List<List<String>> lines = reader.readLines();

        VertexId firstId = rowTensors.get(0).getId();
        VertexId secondId = rowTensors.get(1).getId();

        assertTrue(lines.size() == 2);
        assertTrue(reader.getHeader().equals(Arrays.asList(
            "{" + firstId + "}" + "[0]", "{" + firstId + "}" + "[1]", "{" + firstId + "}" + "[2]", "{" + firstId + "}" + "[3]", "{" + firstId + "}" + "[4]",
            "{" + secondId + "}" + "[0]", "{" + secondId + "}" + "[1]", "{" + secondId + "}" + "[2]", "{" + secondId + "}" + "[3]"))
        );
        assertTrue(lines.get(0).equals(
            Arrays.asList("1.0", "2.0", "3.0", "4.0", "5.0", "5.0", "4.0", "3.0", "2.0")));
        assertTrue(lines.get(1).equals(
            Arrays.asList("2.0", "4.0", "6.0", "8.0", "10.0", "10.0", "8.0", "6.0", "4.0")));

        file.delete();
    }

    @Test
    public void writeColumnOfTensorsToCsv() throws IOException {
        File file = WriteCsv.asColumns(columnTensors).toFile(File.createTempFile("test",".csv"));

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(false);
        List<List<String>> lines = reader.readLines();

        assertTrue(lines.size() == 5);
        assertTrue(lines.get(0).equals(Arrays.asList("1.0", "5.0")));
        assertTrue(lines.get(4).equals(Arrays.asList("5.0", "-")));
        file.delete();
    }

    @Test
    public void writeColumnOfTensorsToCsvWithCustomEmptyValue() throws IOException {
        File file = WriteCsv.asColumns(columnTensors).withEmptyValue("None").toFile(File.createTempFile("test",".csv"));

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(false);
        List<List<String>> lines = reader.readLines();

        assertTrue(lines.size() == 5);
        assertTrue(lines.get(0).equals(Arrays.asList("1.0", "5.0")));
        assertTrue(lines.get(4).equals(Arrays.asList("5.0", "None")));
        file.delete();
    }

    @Test
    public void writeColumnOfIntegerTensorsToCsv() throws IOException {
        File file = WriteCsv.asColumns(integerColumnTensors).toFile(File.createTempFile("test",".csv"));

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(false);
        List<List<String>> lines = reader.readLines();

        assertTrue(lines.size() == 3);
        assertTrue(lines.get(0).equals(Arrays.asList("1", "3")));
        assertTrue(lines.get(2).equals(Arrays.asList("3", "1")));
        file.delete();
    }

    @Test
    public void writeColumnOfTensorsToCsvWithHeader() throws IOException {
        File file = WriteCsv.asColumns(columnTensors).withDefaultHeader().toFile(File.createTempFile("test",".csv"));

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(true);
        List<List<String>> lines = reader.readLines();

        VertexId firstId = columnTensors.get(0).getId();
        VertexId secondId = columnTensors.get(1).getId();

        assertTrue(lines.size() == 5);
        assertTrue(reader.getHeader().equals(Arrays.asList("{" + firstId + "}", "{" + secondId + "}")));
        assertTrue(lines.get(0).equals(Arrays.asList("1.0", "5.0")));
        assertTrue(lines.get(4).equals(Arrays.asList("5.0", "-")));
        file.delete();
    }

    @Test
    public void writeRowOfTensorsToCsv() throws IOException {
        File file = WriteCsv.asRows(rowTensors).toFile(File.createTempFile("test",".csv"));

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(false);
        List<List<String>> lines = reader.readLines();

        assertTrue(lines.size() == 2);
        assertTrue(lines.get(0).equals(Arrays.asList("1.0", "2.0", "3.0", "4.0", "5.0")));
        assertTrue(lines.get(1).equals(Arrays.asList("5.0", "4.0", "3.0", "2.0", "-")));
        file.delete();
    }

    @Test
    public void writeRowOfTensorsToCsvWithCustomEmptyValue() throws IOException {
        File file = WriteCsv.asRows(rowTensors).withEmptyValue("/").toFile(File.createTempFile("test",".csv"));

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(false);
        List<List<String>> lines = reader.readLines();

        assertTrue(lines.size() == 2);
        assertTrue(lines.get(0).equals(Arrays.asList("1.0", "2.0", "3.0", "4.0", "5.0")));
        assertTrue(lines.get(1).equals(Arrays.asList("5.0", "4.0", "3.0", "2.0", "/")));
        file.delete();
    }

    @Test
    public void writeRowOfTensorsToCsvWithHeader() throws IOException {
        File file = WriteCsv.asRows(rowTensors).withDefaultHeader().toFile(File.createTempFile("test",".csv"));

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(true);
        List<List<String>> lines = reader.readLines();

        assertTrue(lines.size() == 2);
        assertTrue(reader.getHeader().equals(Arrays.asList("[0]", "[1]", "[2]", "[3]", "[4]")));
        assertTrue(lines.get(0).equals(Arrays.asList("1.0", "2.0", "3.0", "4.0", "5.0")));
        assertTrue(lines.get(1).equals(Arrays.asList("5.0", "4.0", "3.0", "2.0", "-")));
        file.delete();
    }

    @Test
    public void writeRowOfScalarsToCsv() throws IOException {
        File file = WriteCsv.asColumns(scalarTensors).toFile(File.createTempFile("test",".csv"));

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(false);
        List<List<String>> lines = reader.readLines();

        assertTrue(lines.size() == 1);
        assertTrue(lines.get(0).equals(Arrays.asList("0.5", "1.5")));
        file.delete();
    }

    @Test
    public void writeRowOfScalarsToCsvWithHeader() throws IOException {
        File file = WriteCsv.asColumns(scalarTensors).withDefaultHeader().toFile(File.createTempFile("test",".csv"));

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(true);
        List<List<String>> lines = reader.readLines();

        VertexId firstId = scalarTensors.get(0).getId();
        VertexId secondId = scalarTensors.get(1).getId();

        assertTrue(lines.size() == 1);
        assertTrue(reader.getHeader().equals(Arrays.asList("{" + firstId + "}", "{" + secondId + "}")));
        assertTrue(lines.get(0).equals(Arrays.asList("0.5", "1.5")));
        file.delete();
    }

    @Test
    public void writeRowOfScalarsToCsvWithCustomHeader() throws IOException {
        String[] customHeader = new String[]{"Temperature", "Humidity"};
        File file = WriteCsv.asColumns(scalarTensors).withHeader(customHeader).toFile(File.createTempFile("test",".csv"));

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(true);
        List<List<String>> lines = reader.readLines();

        assertTrue(lines.size() == 1);
        assertTrue(reader.getHeader().equals(Arrays.asList(customHeader)));
        assertTrue(lines.get(0).equals(Arrays.asList("0.5", "1.5")));
        file.delete();
    }

    @Test
    public void writeRowOfScalarsToCsvWithCustomDelimiter() throws IOException {
        File file = WriteCsv.asColumns(scalarTensors).withSeparator('\t').toFile(File.createTempFile("test",".csv"));

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(false).withDelimiter("\t");
        List<List<String>> lines = reader.readLines();

        assertTrue(lines.size() == 1);
        assertTrue(lines.get(0).equals(Arrays.asList("0.5", "1.5")));
        file.delete();
    }

}
