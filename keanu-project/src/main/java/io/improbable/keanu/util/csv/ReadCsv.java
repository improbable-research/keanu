package io.improbable.keanu.util.csv;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * This class provides static helper functions for loading csv data from
 * various locations.
 */
public class ReadCsv {

    private ReadCsv() {
    }

    /**
     * @param fileOnClassPath location relative to src/main/resources (or just the classpath)
     * @return a reader for the resource file
     */
    public static CsvReader fromResources(String fileOnClassPath) {
        return new CsvReader(() -> getFileFromResources(fileOnClassPath));
    }

    public static CsvReader fromFile(String filePath) {
        return fromFile(Paths.get(filePath));
    }

    /**
     * @param filePath the full path to the file for loading
     * @return a reader for the file
     */
    public static CsvReader fromFile(Path filePath) {
        return new CsvReader(() -> {
            try {
                return new BufferedReader(new FileReader(filePath.toFile()));
            } catch (FileNotFoundException fnfe) {
                throw new UncheckedIOException(fnfe);
            }
        });
    }

    public static CsvReader fromString(String csvString) {
        return new CsvReader(() -> new StringReader(csvString));
    }

    /**
     * This reads a file that is located in the src/main/resources folder. Files located there are by default available
     * on the class path in java. Loading it this way makes sure it can be loaded from any platform from any current
     * directory.
     *
     * @param fileOnClassPath the file located in the resources folder
     * @return a reader of that file
     */
    private static Reader getFileFromResources(String fileOnClassPath) {
        InputStream csvFileStream = ReadCsv.class.getClassLoader().getResourceAsStream(fileOnClassPath);
        if (csvFileStream == null) {
            throw new UncheckedIOException(new FileNotFoundException(fileOnClassPath + " not found on class path!"));
        }
        return new InputStreamReader(csvFileStream);
    }
}
