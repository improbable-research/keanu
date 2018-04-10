package com.example.starter;

import io.improbable.keanu.util.csv.CsvReader;
import io.improbable.keanu.util.csv.ReadCsv;

public class Data {

    public Data() { }

    public static Data load(String fileName) {
        //Load a csv file from src/main/resources
        CsvReader csvReader = ReadCsv.fromResources(fileName).expectHeader(false);

        /**
         * Do something with csv lines here
         */

        //create new Data object from csv
        return new Data();
    }

}
