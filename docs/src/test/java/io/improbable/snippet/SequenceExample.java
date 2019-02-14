package io.improbable.snippet;

import io.improbable.keanu.templating.Sequence;
import io.improbable.keanu.templating.SequenceBuilder;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.csv.ReadCsv;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.util.List;

public class SequenceExample {
    //%%SNIPPET_START%% SequenceData
    public static class MyData {
        public double x;
        public double y;

        public MyData(String x, String y) {
            this.x = Double.parseDouble(x);
            this.y = Double.parseDouble(y);
        }
    }
    //%%SNIPPET_END%% SequenceData
    //%%SNIPPET_START%% Sequence

    /**
     * Each sequence item contains a linear regression model:
     * VertexY = VertexX * VertexM + VertexB
     *
     * @param dataFileName The input data file defines, for each sequence item:
     *                     - the value of the input, VertexX
     *                     - the value of the observed output, VertexY
     */
    public Sequence buildSequence(String dataFileName) {
        //Parse the csv data to MyData objects
        List<MyData> allMyData = ReadCsv.fromFile(dataFileName)
            .asRowsDefinedBy(MyData.class)
            .load();

        DoubleVertex m = new GaussianVertex(0, 1);
        DoubleVertex b = new GaussianVertex(0, 1);
        VertexLabel xLabel = new VertexLabel("x");
        VertexLabel yLabel = new VertexLabel("y");

        //Build sequence from each line in the csv
        Sequence sequence = new SequenceBuilder<MyData>()
            .fromIterator(allMyData.iterator())
            .withFactory((item, csvMyData) -> {

                ConstantDoubleVertex x = new ConstantDoubleVertex(csvMyData.x).setLabel(xLabel);
                DoubleVertex y = m.multiply(x).plus(b).setLabel(yLabel);

                DoubleVertex yObserved = new GaussianVertex(y, 1);
                yObserved.observe(csvMyData.y);

                // this labels the x and y vertex for later use
                item.add(x);
                item.add(y);
            })
            .build();

        //now you have access to the "x" from any one of the sequence
        DoubleTensor valueForXAtCSVLine1 = sequence.asList()
            .get(1) // get sequence item 1 which is built from csv line 1
            .<DoubleVertex>get(xLabel) //get the vertex that we labelled "x" in that item
            .getValue(); //get the value from that vertex

        //Now run an inference algorithm on vertex m and vertex b and you have linear regression

        return sequence;
    }
    //%%SNIPPET_END%% Sequence
}
