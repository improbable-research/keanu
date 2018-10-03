package io.improbable.docs;

import io.improbable.keanu.plating.PlateBuilder;
import io.improbable.keanu.plating.Plates;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.csv.CsvReader;
import io.improbable.keanu.util.csv.ReadCsv;
import io.improbable.keanu.vertices.VertexLabel;
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

  /**
   * Each plate contains a linear regression model:
   *
   * <p>m x \ / * | b | \| +
   *
   * <p>The input data file defines, for each plate: - the value of the constant vertex "x" - the
   * observed value of the vertex "+" (a.k.a. "y")
   *
   * @param dataFileName - the file containing data in (x,y) format
   */
  public Plates buildPlates(String dataFileName) {

    // Read data from a csv file
    CsvReader csvReader = ReadCsv.fromResources(dataFileName);

    // Parse the csv data to MyData objects
    List<MyData> allMyData =
        csvReader
            .streamLines()
            .map(line -> new MyData(line.get(0), line.get(1)))
            .collect(Collectors.toList());

    DoubleVertex m = new GaussianVertex(0, 1);
    DoubleVertex b = new GaussianVertex(0, 1);
    VertexLabel xLabel = new VertexLabel("x");
    VertexLabel yLabel = new VertexLabel("y");

    // Build plates from each line in the csv
    Plates plates =
        new PlateBuilder<MyData>()
            .fromIterator(allMyData.iterator())
            .withFactory(
                (plate, csvMyData) -> {
                  ConstantDoubleVertex x = new ConstantDoubleVertex(csvMyData.x);
                  DoubleVertex y = m.multiply(x).plus(b);

                  DoubleVertex yObserved = new GaussianVertex(y, 1);
                  yObserved.observe(csvMyData.y);

                  // this labels the x and y vertex for later use
                  plate.add(xLabel, x);
                  plate.add(yLabel, y);
                })
            .build();

    // now you have access to the "x" from any one of the plates
    DoubleTensor valueForXAtCSVLine1 =
        plates
            .asList()
            .get(1) // get plate 1 which is build from csv line 1
            .<DoubleVertex>get(xLabel) // get the vertex that we labelled "x" in that plate
            .getValue(); // get the value from that vertex

    // Now run an inference algorithm on vertex m and vertex b and you have linear regression

    return plates;
  }
}
