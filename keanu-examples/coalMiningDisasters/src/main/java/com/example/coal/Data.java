package com.example.coal;

import io.improbable.keanu.util.csv.CsvReader;
import io.improbable.keanu.util.csv.ReadCsv;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Data {

    public int startYear;
    public int endYear;
    public int[] years;
    public int[] disasters;

    public Data(Map<Integer, Integer> yearToDisasterData) {

        startYear = Integer.MAX_VALUE;
        endYear = Integer.MIN_VALUE;
        years = new int[yearToDisasterData.size()];
        disasters = new int[yearToDisasterData.size()];

        int i = 0;
        for (Map.Entry<Integer, Integer> yearToDisaster : yearToDisasterData.entrySet()) {

            disasters[i] = yearToDisaster.getValue();
            int year = yearToDisaster.getKey();
            years[i] = year;

            //find start and end year from data
            startYear = Math.min(startYear, year);
            endYear = Math.max(endYear, year);

            i++;
        }

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
