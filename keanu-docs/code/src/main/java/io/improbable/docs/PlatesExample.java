package io.improbable.docs;

import io.improbable.keanu.plating.PlateBuilder;
import io.improbable.keanu.plating.Plates;
import io.improbable.keanu.util.csv.CsvReader;
import io.improbable.keanu.util.csv.ReadCsv;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.util.List;
import java.util.stream.Collectors;

public class PlatesExample {

    public static class MyData {
        public double x;
        public double y;

        public MyData(String x, String y) {
            this.x = Double.parseDouble(x);
            this.y = Double.parseDouble(y);
        }
    }

    public void buildPlates() {

        //Read data from a csv file
        CsvReader csvReader = ReadCsv.fromFile("./my_file.csv");

        //Parse the csv data to MyData objects
        List<MyData> allMyData = csvReader.streamLines()
            .map(line -> new MyData(line.get(0), line.get(1)))
            .collect(Collectors.toList());

        DoubleVertex m = new GaussianVertex(0, 1);
        DoubleVertex b = new GaussianVertex(0, 1);

        //Build plates from each line in the csv
        Plates plates = new PlateBuilder<MyData>()
            .fromIterator(allMyData.iterator())
            .withFactory((plate, csvMyData) -> {

                ConstantDoubleVertex x = new ConstantDoubleVertex(csvMyData.x);
                DoubleVertex y = m.multiply(x).plus(b);

                DoubleVertex yObserved = new GaussianVertex(y, 1);
                yObserved.observe(csvMyData.y);

                // this labels the x and y vertex for later use
                plate.add("x", x);
                plate.add("y", y);
            })
            .build();

        //now you have access to the "x" from any one of the plates
        double valueForXAtCSVLine1 = plates.asList()
            .get(1) // get plate 1 which is build from csv line 1
            .<Double>get("x") //get the vertex that we labelled "x" in that plate
            .getValue(); //get the value from that vertex

        //Now run an inference algorithm on vertex m and vertex b and you have linear regression

    }
}
