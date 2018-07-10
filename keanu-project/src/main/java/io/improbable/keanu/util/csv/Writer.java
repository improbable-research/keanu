package io.improbable.keanu.util.csv;

import com.opencsv.CSVWriter;
import com.opencsv.ICSVWriter;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.List;

public abstract class Writer {

    private char separator = ICSVWriter.DEFAULT_SEPARATOR;
    private char quoteChar = ICSVWriter.DEFAULT_QUOTE_CHARACTER;
    private char escapeChar= ICSVWriter.DEFAULT_ESCAPE_CHARACTER;
    private String lineEnd = ICSVWriter.DEFAULT_LINE_END;
    private String[] header = null;
    private boolean headerEnabled = false;

    abstract File toFile(String file);

    abstract Writer withDefaultHeader();

    public Writer withSeparator(char s) {
        separator = s;
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

    public Writer withHeader(String[] h) {
        header = h;
        withHeaderEnabled(true);
        return this;
    }

    File writeToFile(String file, List<String[]> data) {
        return writeToFile(new File(file), data);
    }

    File writeToFile(File file, List<String[]> data) {
        return writeToFile(file, data, separator, quoteChar, escapeChar, lineEnd);
    }

    File writeToFile(String file, List<String[]> data, char separator, char quoteChar, char escapeChar, String lineEnd) {
        return writeToFile(new File(file), data, separator, quoteChar, escapeChar, lineEnd);
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
}
