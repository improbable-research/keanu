package com.example.starter;

import io.improbable.keanu.util.csv.ReadCsv;

import java.util.List;

/**
 * This class contains the entire contents of the data_example.csv file.
 * There is a inner class Point that represents a line in the csv file.
 * There is also a static method load() that can be used to create a Data
 * object from a specified file.
 */
public class Data {

    public List<CsvLine> csvLines;

    public Data(List<CsvLine> csvLines) {
        this.csvLines = csvLines;
    }

    /**
     * An example class to load the csv lines into.
     * This expects the csv to have column labels of
     * mu and sigma.
     */
    public static class CsvLine {
        public double mu;
        public double sigma;
    }

    /**
     * @param fileName the name of the csv file in the resource folder
     * @return a Data object with the contents of the csv file
     */
    public static Data load(String fileName) {

        //Load a csv file from src/main/resources
        List<CsvLine> csvLines = ReadCsv
            .fromResources(fileName)
            .expectHeader(true)
            .as(CsvLine.class)
            .asList();

        //create new Data object from csv
        return new Data(csvLines);
    }

}
