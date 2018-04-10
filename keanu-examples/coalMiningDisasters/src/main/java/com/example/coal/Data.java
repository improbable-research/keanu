package com.example.coal;

import io.improbable.keanu.util.csv.CsvReader;
import io.improbable.keanu.util.csv.ReadCsv;

import java.util.HashMap;
import java.util.IntSummaryStatistics;
import java.util.List;
import java.util.Map;

public class Data {

    public Map<Integer, Integer> yearToDisasterCounts;
    public int startYear;
    public int endYear;

    public Data(Map<Integer, Integer> yearToDisasterCount) {
        this.yearToDisasterCounts = yearToDisasterCount;

        //find start and end year from data
        IntSummaryStatistics yearStats = yearToDisasterCounts.keySet().stream().mapToInt(i -> i).summaryStatistics();
        startYear = yearStats.getMin();
        endYear = yearStats.getMax();
    }

    public static Data load(String fileName) {
        //Load a csv file from src/main/resources
        CsvReader csvReader = ReadCsv.fromResources(fileName).expectHeader(false);

        Map<Integer, Integer> yearToDisasterCounts = new HashMap<>();
        for (List<String> csvLine : csvReader.readLines()) {
            // parses lines e.g. "1851,4"
            yearToDisasterCounts.put(Integer.parseInt(csvLine.get(0)), Integer.parseInt(csvLine.get(1)));
        }

        return new Data(yearToDisasterCounts);
    }

}
