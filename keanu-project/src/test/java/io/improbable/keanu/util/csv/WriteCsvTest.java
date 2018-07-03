package io.improbable.keanu.util.csv;

import io.improbable.keanu.algorithms.NetworkSamples;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertTrue;

public class WriteCsvTest {

    private NetworkSamples samples;

    @Before
    public void setup() {

        Map<Long, List<Integer>> sampleMap = new HashMap<>();
        sampleMap.put(1L, Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
        sampleMap.put(2L, Arrays.asList(9, 8, 7, 6, 5, 4, 3, 2, 1, 0));

        samples = new NetworkSamples(sampleMap, 10);
    }

    @Test
    public void writeCsv() {
        File file = WriteCsv.allSamples(samples, 1L, 2L).toFile("test", "../");

        CsvReader reader = ReadCsv.fromFile(file.toPath()).expectHeader(false);
        List<List<String>> lines = reader.readLines();

        assertTrue(lines.size() == 10);
        assertTrue(reader.readLines().get(0).equals(Arrays.asList(new String[]{"1", "9"})));
        assertTrue(reader.readLines().get(9).equals(Arrays.asList(new String[]{"10", "0"})));

        file.delete();
    }

}
