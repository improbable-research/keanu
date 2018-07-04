package io.improbable.keanu.util.csv;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class CsvWriter {

    private static final String DEFAULT_DELIMITER = ",";
    private static final String DEFAULT_SUFFIX = ".csv";
    private static final String NEWLINE = "\n";

    private final List<List<String>> data;
    private String delimiter;
    private boolean writeHeader;
    private List<String> header;
    private FileWriter fileWriter;

    public CsvWriter(List<List<String>> data) {
        this.data = data;
        this.delimiter = DEFAULT_DELIMITER;
        this.writeHeader = false;
        this.header = null;
        this.fileWriter = null;
    }

    public File toFile(String filename, String directory) {
        return toFile(filename, new File(directory));
    }

    public File toFile(String filename, File directory) {
        try {
            return toFile(File.createTempFile(filename, DEFAULT_SUFFIX, directory));
        } catch (IOException e) {
            throw new IllegalArgumentException("Could not create: " + filename + " at: " + directory + ". " + e);
        }
    }

    public File toFile(File file) {
        try {
            fileWriter = new FileWriter(file);
            writeHeader();

            for (List<String> row : data) {
                int count = 0;
                for (String dataPoint : row) {
                    fileWriter.append(dataPoint);
                    if (count != row.size() - 1) {
                        fileWriter.append(delimiter);
                    }
                    count++;
                }
                fileWriter.append(NEWLINE);
            }
        } catch (Exception e) {
            throw new IllegalStateException("Failed while writing to: " + file + ". " + e);
        } finally {
            try {
                fileWriter.flush();
                fileWriter.close();
            } catch (IOException e) {
                throw new IllegalStateException("Failed while closing File Writer. " + e);
            }
        }
        return file;
    }

    private void writeHeader() throws IOException {
        if (writeHeader) {
            String trimmedHeader = header.toString().substring(1, header.toString().length() - 1);
            fileWriter.append(trimmedHeader);
            fileWriter.append(NEWLINE);
        }
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



}
