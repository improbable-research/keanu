package io.improbable.keanu.util.csv;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.util.*;

import static org.junit.Assert.assertTrue;

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

        Map<Long, List<DoubleTensor>> networkSamples = new HashMap<>();
        networkSamples.put(g1.getId(), Arrays.asList(g1.getValue(), g1.times(2).getValue()));
        networkSamples.put(g2.getId(), Arrays.asList(g2.getValue(), g2.times(2).getValue()));
        samples = new NetworkSamples(networkSamples, 2);
    }

    @Test
    public void writeSamplesToCsv() {
        File file = WriteCsv.asSamples(samples, rowTensors).toFile("test", "../");

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(false);
        List<List<String>> lines = reader.readLines();

        assertTrue(lines.size() == 2);
        assertTrue(reader.readLines().get(0).equals(
            Arrays.asList("1.0", "2.0", "3.0", "4.0", "5.0", "5.0", "4.0", "3.0", "2.0")));
        assertTrue(reader.readLines().get(1).equals(
            Arrays.asList("2.0", "4.0", "6.0", "8.0", "10.0", "10.0", "8.0", "6.0", "4.0")));

        file.delete();
    }

    @Test
    public void writeSamplesToCsvWithHeader() {
        File file = WriteCsv.asSamples(samples, rowTensors).withDefaultHeader().toFile("test", "../");

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(true);
        List<List<String>> lines = reader.readLines();

        long firstId = rowTensors.get(0).getId();
        long secondId = rowTensors.get(1).getId();

        assertTrue(lines.size() == 2);
        assertTrue(reader.getHeader().equals(Arrays.asList(
            firstId + "[0]", firstId + "[1]", firstId + "[2]", firstId + "[3]", firstId + "[4]",
            secondId + "[0]", secondId + "[1]", secondId + "[2]", secondId + "[3]"))
        );
        assertTrue(reader.readLines().get(0).equals(
            Arrays.asList("1.0", "2.0", "3.0", "4.0", "5.0", "5.0", "4.0", "3.0", "2.0")));
        assertTrue(reader.readLines().get(1).equals(
            Arrays.asList("2.0", "4.0", "6.0", "8.0", "10.0", "10.0", "8.0", "6.0", "4.0")));

        file.delete();
    }

    @Test
    public void writeColumnOfTensorsToCsv() {
        File file = WriteCsv.asColumns(columnTensors).toFile("test", "../");

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(false);
        List<List<String>> lines = reader.readLines();

        assertTrue(lines.size() == 5);
        assertTrue(reader.readLines().get(0).equals(Arrays.asList("1.0", "5.0")));
        assertTrue(reader.readLines().get(4).equals(Arrays.asList("5.0", "-")));
        file.delete();
    }

    @Test
    public void writeColumnOfIntegerTensorsToCsv() {
        File file = WriteCsv.asColumns(integerColumnTensors).toFile("test", "../");

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(false);
        List<List<String>> lines = reader.readLines();

        assertTrue(lines.size() == 3);
        assertTrue(reader.readLines().get(0).equals(Arrays.asList("1", "3")));
        assertTrue(reader.readLines().get(2).equals(Arrays.asList("3", "1")));
        file.delete();
    }

    @Test
    public void writeColumnOfTensorsToCsvWithHeader() {
        File file = WriteCsv.asColumns(columnTensors).withDefaultHeader().toFile("test", "../");

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(true);
        List<List<String>> lines = reader.readLines();

        long firstId = columnTensors.get(0).getId();
        long secondId = columnTensors.get(1).getId();

        assertTrue(lines.size() == 5);
        assertTrue(reader.getHeader().equals(Arrays.asList(new String[]{String.valueOf(firstId), String.valueOf(secondId)})));
        assertTrue(reader.readLines().get(0).equals(Arrays.asList("1.0", "5.0")));
        assertTrue(reader.readLines().get(4).equals(Arrays.asList("5.0", "-")));
        file.delete();
    }


    @Test
    public void writeRowOfTensorsToCsv() {
        File file = WriteCsv.asRows(rowTensors).toFile("test", "../");

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(false);
        List<List<String>> lines = reader.readLines();

        assertTrue(lines.size() == 2);
        assertTrue(reader.readLines().get(0).equals(Arrays.asList("1.0", "2.0", "3.0", "4.0", "5.0")));
        assertTrue(reader.readLines().get(1).equals(Arrays.asList("5.0", "4.0", "3.0", "2.0", "-")));
        file.delete();

    }

    @Test
    public void writeRowOfTensorsToCsvWithHeader() {
        File file = WriteCsv.asRows(rowTensors).withDefaultHeader().toFile("test", "../");

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(true);
        List<List<String>> lines = reader.readLines();

        assertTrue(lines.size() == 2);
        assertTrue(reader.getHeader().equals(Arrays.asList("[0]", "[1]", "[2]", "[3]", "[4]")));
        assertTrue(reader.readLines().get(0).equals(Arrays.asList("1.0", "2.0", "3.0", "4.0", "5.0")));
        assertTrue(reader.readLines().get(1).equals(Arrays.asList("5.0", "4.0", "3.0", "2.0", "-")));
        file.delete();

    }

    @Test
    public void writeRowOfScalarsToCsv() {
        File file = WriteCsv.asColumns(scalarTensors).toFile("test", "../");

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(false);
        List<List<String>> lines = reader.readLines();

        assertTrue(lines.size() == 1);
        assertTrue(reader.readLines().get(0).equals(Arrays.asList("0.5", "1.5")));
        file.delete();

    }

    @Test
    public void writeRowOfScalarsToCsvWithHeader() {
        File file = WriteCsv.asColumns(scalarTensors).withDefaultHeader().toFile("test", "../");

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(true);
        List<List<String>> lines = reader.readLines();

        long firstId = scalarTensors.get(0).getId();
        long secondId = scalarTensors.get(1).getId();

        assertTrue(lines.size() == 1);
        assertTrue(reader.getHeader().equals(Arrays.asList(new String[]{String.valueOf(firstId), String.valueOf(secondId)})));
        assertTrue(reader.readLines().get(0).equals(Arrays.asList("0.5", "1.5")));
        file.delete();

    }

    @Test
    public void writeRowOfScalarsToCsvWithCustomHeader() {
        List<String> customHeader = Arrays.asList("Temperature", "Humidity");
        File file = WriteCsv.asColumns(scalarTensors).withCustomHeader(customHeader).toFile("test", "../");

        CsvReader reader = ReadCsv.fromFile(file).expectHeader(true);
        List<List<String>> lines = reader.readLines();

        assertTrue(lines.size() == 1);
        assertTrue(reader.getHeader().equals(customHeader));
        assertTrue(reader.readLines().get(0).equals(Arrays.asList("0.5", "1.5")));
        file.delete();

    }

}
