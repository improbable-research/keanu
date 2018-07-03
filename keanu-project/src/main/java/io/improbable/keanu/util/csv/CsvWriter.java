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
            e.printStackTrace();
            return null;
        }
    }

    public File toFile(File file) {
        try {
            fileWriter = new FileWriter(file);
            writeHeader();

            for (List<String> row : data) {
                for (String dataPoint : row) {
                    fileWriter.append(dataPoint);
                    fileWriter.append(delimiter);
                }
                fileWriter.append(NEWLINE);
            }

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                fileWriter.flush();
                fileWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return file;
    }

    private void writeHeader() throws IOException {
        if (writeHeader) {
            fileWriter.append(header.toString());
            fileWriter.append(NEWLINE);
        }
    }

}
