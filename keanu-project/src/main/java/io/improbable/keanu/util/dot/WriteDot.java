package io.improbable.keanu.util.dot;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Set;

public class WriteDot {

    public static void outputDot(String fileName, BayesianNetwork net) {
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));

            writer.write("digraph TestGraph {");
            writer.newLine();

            for (Vertex v : net.getVertices()) {

                Set children  = v.getChildren();
                for (Object child : children) {
                    // TODO labels are not necessarily set
                    writer.write("<" + v.getLabel() + "> -> <" + ((Vertex) child).getLabel() + ">");
                    writer.newLine();
                }
            }

            writer.write("}");

            writer.close();

        } catch (IOException e) {
            System.out.println("Could not write to the file.");
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        writerTest();
    }

    public static void writerTest() {
        Collection<? extends Vertex> vertices;

        double mu = 0;
        double sigma = 0;
        GaussianVertex v1 = new GaussianVertex(mu, sigma);
        GaussianVertex v2 = new GaussianVertex(mu, sigma);
        GaussianVertex v3 = new GaussianVertex(mu, sigma);

        DoubleVertex v4 = v1.multiply(v2);
        DoubleVertex v5 = v3.plus(v4);

        BayesianNetwork myNet = new BayesianNetwork(v4.getConnectedGraph());

        outputDot("TestFile.txt", myNet);

    }
}
