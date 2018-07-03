package io.improbable.keanu.util.csv;

import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertTrue;

public class CsvWriterTest {

    private List<List<String>> data = new ArrayList<>();
    private String[] row1 = new String[]{"a", "b", "c"};
    private String[] row2 = new String[]{"a", "b", "c", "d"};

    @Before
    public void initData() {
        data.add(Arrays.asList(row1));
        data.add(Arrays.asList(row2));
    }

    @Test
    public void canWriteFile() {
        CsvWriter writer = new CsvWriter(data);
        File file = writer.toFile("test", "../");

        CsvReader reader = ReadCsv.fromFile(file.toPath()).expectHeader(false);
        List<List<String>> lines = reader.readLines();

        assertTrue(lines.size() == 2);
        assertTrue(reader.readLines().get(0).equals(Arrays.asList(row1)));
        assertTrue(reader.readLines().get(1).equals(Arrays.asList(row2)));

        file.delete();
    }
}
