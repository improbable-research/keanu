package io.improbable.snippet;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.nuts.NUTS;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.util.csv.CsvReader;
import io.improbable.keanu.util.csv.ReadCsv;
import io.improbable.keanu.util.csv.WriteCsv;
import io.improbable.keanu.util.csv.pojo.CsvProperty;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import java.io.File;
import java.util.List;
import java.util.stream.Stream;

public class CsvOperations {

    public static void main(String[] args) {
//%%SNIPPET_START%% ReadPOJOIn
List<MyClass> myPojos = ReadCsv.fromFile("some/file/path")
        .asRowsDefinedBy(MyClass.class)
        .load();
//%%SNIPPET_END%% ReadPOJOIn
    }

//%%SNIPPET_START%% ReadPOJOObj
public class MyClass {
    public String myString;
    public int myInt;
}
//%%SNIPPET_END%% ReadPOJOObj

//%%SNIPPET_START%% ReadSetter
public class MySetterClass {
    private String myString;

    public void setMyString(String aString) {
        myString = aString;
    }
}
//%%SNIPPET_END%% ReadSetter

    private static void readCsvs() {
//%%SNIPPET_START%% ReadFromCurrent
CsvReader reader1 = ReadCsv.fromFile("~/my_filename.csv");
//%%SNIPPET_END%% ReadFromCurrent
//%%SNIPPET_START%% ReadFromResource
CsvReader reader2 = ReadCsv.fromResources("my_other_filename.csv");
//%%SNIPPET_END%% ReadFromResource

//%%SNIPPET_START%% ReadLines
for (List<String> csvLine : reader1.readLines()) {
    //do something with your csv line
}
//%%SNIPPET_END%% ReadLines

//%%SNIPPET_START%% ReadStream
try (Stream<List<String>> lineStream = reader2.streamLines()) {

    lineStream.forEach(line -> {
        //do something with your line.
    });
}
//%%SNIPPET_END%% ReadStream
    }

    private static List<DoubleVertex> runMyModel() {
        return (null);
    }

    private static void writeCsvs() {
//%%SNIPPET_START%% WriteVars
List<DoubleVertex> inferredVariables = runMyModel();

inferredVariables.get(0); // [0.5, 1.0, 1.5, 2.0]
inferredVariables.get(1); // [5.0, 10.0, 15.0, 20.0]
inferredVariables.get(2); // [50.0, 100.0, 150.0]
//%%SNIPPET_END%% WriteVars

//%%SNIPPET_START%% WriteColumnTest
File file = WriteCsv.asColumns(inferredVariables).toFile("columnTest.csv");
//%%SNIPPET_END%% WriteColumnTest

//%%SNIPPET_START%% WriteRowTest
File file2 = WriteCsv.asRows(inferredVariables).toFile("rowTest.csv");
//%%SNIPPET_END%% WriteRowTest

//%%SNIPPET_START%% WriteDefaultHeaders
File file3 = WriteCsv.asColumns(inferredVariables).withDefaultHeader().toFile("columnTest.csv");
//%%SNIPPET_END%% WriteDefaultHeaders

//%%SNIPPET_START%% WriteDefaultHeadersRows
File file4 = WriteCsv.asRows(inferredVariables).withDefaultHeader().toFile("rowTest.csv");
//%%SNIPPET_END%% WriteDefaultHeadersRows

DoubleVertex aVertex = null;
//%%SNIPPET_START%% WriteGetSamples
NetworkSamples samples = NUTS.withDefaultConfig().getPosteriorSamples(
        new BayesianNetwork(aVertex.getConnectedGraph()),
        aVertex,
        1000
);
//%%SNIPPET_END%% WriteGetSamples

//%%SNIPPET_START%% WriteSamples
File file5 = WriteCsv.asSamples(samples, inferredVariables).withDefaultHeader().toFile("test.csv");
//%%SNIPPET_END%% WriteSamples
    }

//%%SNIPPET_START%% ReadProblemHeader
public class MyOtherClass {
    @CsvProperty("my-problematic*header")
    public String myString;
    public int myInt;
}
//%%SNIPPET_END%% ReadProblemHeader
}
