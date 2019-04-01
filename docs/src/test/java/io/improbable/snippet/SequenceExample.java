package io.improbable.snippet;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.templating.Sequence;
import io.improbable.keanu.templating.SequenceBuilder;
import io.improbable.keanu.templating.SequenceItem;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.csv.ReadCsv;
import io.improbable.keanu.vertices.SimpleVertexDictionary;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexDictionary;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.util.List;
import java.util.function.Consumer;

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

    public Sequence buildSequenceTimeSeries() {
        //%%SNIPPET_START%% SequenceTimeSeries
        DoubleVertex two = new ConstantDoubleVertex(2);

        // Define the labels of vertices we will use in our Sequence
        VertexLabel x1Label = new VertexLabel("x1");
        VertexLabel x2Label = new VertexLabel("x2");

        // Define labels for the Proxy Vertices which stand in for a Vertex from the previous SequenceItem.
        // They will be automatically wired up when you construct the Sequence.
        // i.e. these are the 'inputs' to our SequenceItem
        VertexLabel x1InputLabel = SequenceBuilder.proxyFor(x1Label);
        VertexLabel x2InputLabel = SequenceBuilder.proxyFor(x2Label);

        // Define a factory method that creates proxy vertices using the proxy vertex labels and then uses these
        // to define the computation graph of the Sequence.
        // Note we have labeled the output vertices of this SequenceItem
        Consumer<SequenceItem> factory = sequenceItem -> {
            DoubleProxyVertex x1Input = new DoubleProxyVertex(x1InputLabel);
            DoubleProxyVertex x2Input = new DoubleProxyVertex(x2InputLabel);

            DoubleVertex x1Output = x1Input.multiply(two).setLabel(x1Label);
            DoubleVertex x2Output = x2Input.plus(x1Output).setLabel(x2Label);

            sequenceItem.addAll(x1Input, x2Input, x1Output, x2Output);
        };

        // Create the starting values of our sequence
        DoubleVertex x1Start = new ConstantDoubleVertex(4).setLabel(x1Label);
        DoubleVertex x2Start = new ConstantDoubleVertex(4).setLabel(x2Label);
        VertexDictionary dictionary = SimpleVertexDictionary.of(x1Start, x2Start);

        Sequence sequence = new SequenceBuilder<Integer>()
            .withInitialState(dictionary)
            .named("Keanu-Example")
            .count(5)
            .withFactory(factory)
            .build();

        // We can now put all the vertices in the sequence into a Bayes Net:
        BayesianNetwork network = sequence.toBayesianNetwork();

        // Within `network` our vertices will have the labels of the form:
        // Keanu-Example.Sequence_Item_<<index>>.<<hash>>.<<vertex-label>>
        // where the <<hash>> is a unique identifier for the Sequence.
        // You can get all the vertices with a particular name, regardless of which SequenceItem they belong to.
        List<Vertex> allXVertices = network.getVerticesIgnoringNamespace(x1Label.getUnqualifiedName());

        // You get vertices from specific sequence items
        // For instance here we retrieve a vertex from the last sequence item
        Vertex x1Retrieved = sequence.getLastItem().get(x1Label);

        // Or you can iterate over all the sequence items using an iterator
        for (SequenceItem item : sequence) {
            Vertex x2Retrieved = item.get(x2Label);
        }

        // Or you can get the SequenceItem as a list to retrieve an item at a specific index
        List<SequenceItem> sequenceItems = sequence.asList();
        SequenceItem secondSequenceItem = sequenceItems.get(1);
        Vertex x2InSecondSequenceItem = secondSequenceItem.get(x2Label);

        // Finally, you may need to use the save/load interface on `network` and will need a way of accessing timesteps
        // without having access to the `sequence` object
        x1Retrieved = network.getVerticesInNamespace("Keanu-Example", "Sequence_Item_0").get(0);

        //%%SNIPPET_END%% SequenceTimeSeries

        return sequence;
    }
}
