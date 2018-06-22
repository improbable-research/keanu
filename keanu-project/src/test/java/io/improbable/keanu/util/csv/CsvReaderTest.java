package io.improbable.keanu.util.csv;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertTrue;

public class CsvReaderTest {

    String csv = "a,b,c\nhel,lo,world";
    String tsv = "a\tb\tc\nhel\tlo\tworld";

    List<String> line0 = Arrays.asList("a", "b", "c");
    List<String> line1 = Arrays.asList("hel", "lo", "world");

    @Test
    public void givenCsvStringWithHeaderThenReturnsDataAndHeader() {
        CsvReader csvReader = ReadCsv.fromString(csv);
        testWithHeader(csvReader);
    }

    @Test
    public void givenCsvStringWithoutHeaderThenReturnsData() {
        CsvReader csvReader = ReadCsv.fromString(csv).expectHeader(false);
        testWithoutHeader(csvReader);
    }

    @Test
    public void givenTsvStringWithHeaderThenReturnsDataAndHeader() {
        CsvReader csvReader = ReadCsv.fromString(tsv).withDelimiter("\t");
        testWithHeader(csvReader);
    }

    @Test
    public void givenTsvStringWithoutHeaderThenReturnsData() {
        CsvReader csvReader = ReadCsv.fromString(tsv)
            .expectHeader(false)
            .withDelimiter("\t");
        testWithoutHeader(csvReader);
    }

    private void testWithHeader(CsvReader csvReader) {
        List<String> header = csvReader.getHeader();
        List<List<String>> lines = csvReader.readLines();

        assertTrue(header.equals(line0));
        assertTrue(lines.size() == 1);
        assertTrue(lines.get(0).equals(line1));
    }

    private void testWithoutHeader(CsvReader csvReader) {
        List<List<String>> lines = csvReader.readLines();

        assertTrue(lines.size() == 2);
        assertTrue(lines.get(0).equals(line0));
        assertTrue(lines.get(1).equals(line1));
    }
}
