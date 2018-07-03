package io.improbable.keanu.util.csv;

import io.improbable.keanu.algorithms.NetworkSamples;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class WriteCsv {

    private WriteCsv() {
    }

    public static CsvWriter allSamples(NetworkSamples samples, Long... vertexIds) {
        List<List<String>> data = new ArrayList<>();
        Map<Long, List> vertexSamples = new HashMap<>();

        for (Long id : vertexIds) {
            vertexSamples.put(id, samples.get(id).asList());
        }

        int sampleCount = vertexSamples.get(vertexIds[0]).size();

        for (int i = 0; i < sampleCount; i++) {
            List<String> row = new ArrayList<>();
            for (Long id : vertexSamples.keySet()) {
                row.add(vertexSamples.get(id).get(i).toString());
            }
            data.add(row);
        }

        return new CsvWriter(data);
    }

}
