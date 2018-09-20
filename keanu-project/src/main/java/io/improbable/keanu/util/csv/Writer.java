package io.improbable.keanu.util.csv;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import com.opencsv.CSVWriter;
import com.opencsv.ICSVWriter;

public abstract class Writer {

    public static final char DEFAULT_SEPARATOR = ICSVWriter.DEFAULT_SEPARATOR;
    public static final char DEFAULT_QUOTE_CHAR = ICSVWriter.DEFAULT_QUOTE_CHARACTER;
    public static final char DEFAULT_ESCAPE_CHAR = ICSVWriter.DEFAULT_ESCAPE_CHARACTER;
    public static final String DEFAULT_LINE_END = ICSVWriter.DEFAULT_LINE_END;
    public static final String DEFAULT_EMPTY_VALUE = "-";

    private char separator = DEFAULT_SEPARATOR;
    private char quoteChar = DEFAULT_QUOTE_CHAR;
    private char escapeChar = DEFAULT_ESCAPE_CHAR;
    private String lineEnd = DEFAULT_LINE_END;
    private String emptyValue = DEFAULT_EMPTY_VALUE;
    private String[] header = null;
    private boolean headerEnabled = false;

    public abstract File toFile(File file);

    public File toFile(String file) {
        return toFile(new File(file));
    }

    public abstract Writer withDefaultHeader();

    public Writer withSeparator(char s) {
        separator = s;
        return this;
    }

    public Writer withEmptyValue(String emptyValue) {
        this.emptyValue = emptyValue;
        return this;
    }

    public Writer withQuoteCharacter(char q) {
        quoteChar = q;
        return this;
    }

    public Writer withEscapeCharacter(char e) {
        escapeChar = e;
        return this;
    }

    public Writer withLineEnd(String l) {
        lineEnd = l;
        return this;
    }

    public Writer withHeaderEnabled(boolean isEnabled) {
        headerEnabled = isEnabled;
        return this;
    }

    public char getSeparator() {
        return separator;
    }

    public char getQuoteChar() {
        return quoteChar;
    }

    public char getEscapeChar() {
        return escapeChar;
    }

    public String getLineEnd() {
        return lineEnd;
    }

    public String getEmptyValue() {
        return emptyValue;
    }

    File writeToFile(File file, List<String[]> data) {
        return writeToFile(file, data, separator, quoteChar, escapeChar, lineEnd);
    }

    File writeToFile(File file, List<String[]> data, char separator, char quoteChar, char escapeChar, String lineEnd) {
        try (CSVWriter writer = new CSVWriter(new FileWriter(file), separator, quoteChar, escapeChar, lineEnd)) {
            if (headerEnabled) {
                writer.writeNext(header, false);
            }
            writer.writeAll(data, false);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return file;
    }

    String[] createHeader(int size, String headerStyle, Function<Integer, String> func) {
        String[] header = new String[size];
        for (int i = 0; i < size; i++) {
            header[i] = String.format(headerStyle, func.apply(i));
        }
        return header;
    }

    public Writer withHeader(String... h) {
        header = Arrays.copyOf(h, h.length);
        withHeaderEnabled(true);
        return this;
    }

}
