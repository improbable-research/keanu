package io.improbable.keanu.util.csv;

import io.improbable.keanu.util.csv.pojo.bycolumn.ColumnsVectorizedObjectParser;
import io.improbable.keanu.util.csv.pojo.byrow.RowsAsObjectParser;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.util.Collections.emptyList;

public class CsvReader {

    private static final String DEFAULT_DELIMITER = ",";

    private final Supplier<Reader> sourceSupplier;
    private String delimiter;
    private boolean expectHeader;
    private List<String> header;

    public CsvReader(Supplier<Reader> sourceSupplier) {
        this.sourceSupplier = sourceSupplier;
        this.delimiter = DEFAULT_DELIMITER;
        this.expectHeader = true;
        this.header = null;
    }

    /**
     * This will read the entire csv file and return it as a list.
     * Warning: avoid using this for large csv files.
     *
     * @return a list lines that are represented as a list of strings where each
     * string is a field in that line.
     */
    public List<List<String>> readLines() {
        try (Stream<List<String>> stream = streamLines()) {
            return stream.collect(Collectors.toList());
        }
    }

    /**
     * Read all lines from the csv file one at a time without ever holding
     * the entire csv file in memory
     *
     * @return a stream of lines represented by a list of strings where
     * each string is a field in the line.
     */
    public Stream<List<String>> streamLines() {

        BufferedReader bufferedReader = new BufferedReader(sourceSupplier.get());

        if (expectHeader) {
            try {
                header = splitLine(bufferedReader.readLine());
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        return bufferedReader.lines()
            .map(this::splitLine);
    }

    public <T> RowsAsObjectParser<T> asRowsDefinedBy(Class<T> clazz) {
        //Header required for pojo deserialize
        expectHeader(true);
        return new RowsAsObjectParser<>(clazz, streamLines(), getHeader());
    }

    public <T> ColumnsVectorizedObjectParser<T> asVectorizedColumnsDefinedBy(Class<T> clazz) {
        //Header required for pojo deserialize
        expectHeader(true);
        return new ColumnsVectorizedObjectParser<>(clazz, streamLines(), getHeader());
    }

    /**
     * Gets the header of the csv file if one exist otherwise an empty list.
     *
     * @return the header
     */
    public List<String> getHeader() {

        if (!expectHeader) {
            return emptyList();
        }

        if (header == null) {
            try (BufferedReader bufferedReader = new BufferedReader(sourceSupplier.get())) {
                this.header = splitLine(bufferedReader.readLine());
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        return header;
    }

    /**
     * Tells the reader to either treat the first line of the file as a header or as data.
     *
     * @param expectHeader true for first line is header, false for data
     * @return this reader
     */
    public CsvReader expectHeader(boolean expectHeader) {
        this.expectHeader = expectHeader;
        return this;
    }

    /**
     * Defaults to a comma "," but could be \t for tab separated files or something else.
     *
     * @param delimiter the delimiter to use
     * @return this reader
     */
    public CsvReader withDelimiter(String delimiter) {
        this.delimiter = delimiter;
        return this;
    }

    private List<String> splitLine(String line) {
        return Arrays.stream(line.split(delimiter))
            .map(String::trim)
            .collect(Collectors.toList());
    }
}