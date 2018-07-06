package io.improbable.keanu.util.csv;

import java.io.*;
import java.util.List;
import java.util.stream.Collectors;

public class CsvWriter {

    private static final String DEFAULT_DELIMITER = ",";
    private static String DEFAULT_EMPTY_VALUE = "-";
    private static final String NEWLINE = System.lineSeparator();

    private final List<List<String>> data;
    private List<String> header;
    private String delimiter;
    private boolean writeHeader;
    private String emptyValue;

    public CsvWriter(List<List<String>> data, String emptyValue) {
        this.data = data;
        this.delimiter = DEFAULT_DELIMITER;
        this.writeHeader = false;
        this.header = null;
        this.emptyValue = emptyValue;
    }

    public CsvWriter(List<List<String>> data) {
        this(data, DEFAULT_EMPTY_VALUE);
    }

    public File toFile(String filename, String directory) {
        return toFile(filename, new File(directory));
    }

    public File toFile(String filename, File directory) {
        return toFile(directory.toPath().resolve(filename).toFile());
    }

    public File toFile(File file) {
        try (BufferedWriter fileWriter = new BufferedWriter(new FileWriter(file))) {
            if (writeHeader) {
                writeHeader(fileWriter, delimiter);
            }

            for (List<String> row : data) {
                fileWriter.append(toCsvLine(row, delimiter));
                fileWriter.newLine();
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return file;
    }

    private void writeHeader(BufferedWriter fileWriter, String delimiter) throws IOException {
        String csvHeader = toCsvLine(header, delimiter);
        fileWriter.append(csvHeader);
        fileWriter.append(NEWLINE);
    }

    private String toCsvLine(List<String> tokens, String delimiter) {
        return tokens.stream().collect(Collectors.joining(delimiter));
    }

    public CsvWriter withCustomHeader(List<String> header) {
        this.header = header;
        writeHeader = true;
        return this;
    }

    public CsvWriter withDefaultHeader() {
        writeHeader = true;
        return this;
    }

    public CsvWriter disableHeader() {
        writeHeader = false;
        return this;
    }

    public CsvWriter enableHeader() {
        writeHeader = true;
        return this;
    }

    public CsvWriter withCustomDelimiter(String delimiter) {
        this.delimiter = delimiter;
        return this;
    }

    public CsvWriter withCustomEmptyValue(String emptyValue) {
        for (List<String> row : data) {
            for (String value : row) {
                row.set(row.indexOf(value), value.replace(this.emptyValue, emptyValue));
            }
        }
        return this;
    }

}
