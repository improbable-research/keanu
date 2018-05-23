package com.example.starter;


import io.improbable.keanu.algorithms.variational.GradientOptimizer;
import io.improbable.keanu.network.BayesNetDoubleAsContinuous;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.util.Random;

/**
 * This is a simple example of using Keanu that is intended to be used
 * as a starting point for new Keanu projects. It loads data from
 * a csv file found in the resources folder and uses it to build
 * a simple model where we have a prior belief of two random variables
 * and we noisily observe their sum. The model prints out the most
 * probable values for the two random variables given the observation
 * and the most probable true value of their sum given the prior belief.
 */
public class Model {

    public static void main(String[] args) {

        //Load data from a csv file in the resources folder
        Data data = Data.load("data_example.csv");

        //create my model using the data
        Model model = new Model(data);
        model.run();
    }

    private final Data data;

    public double results;

    Model(Data data) {
        this.data = data;
    }

    void run() {

        //Create a random and set its seed if you want your model to run the same each time
        Random random = new Random(1);

        //Use lines from your csv data file
        Data.CsvLine firstCsvLine = data.csvLines.get(0);
        Data.CsvLine secondCsvLine = data.csvLines.get(1);

        //Create your model as a bayesian network
        DoubleVertex A = new GaussianVertex(firstCsvLine.mu, firstCsvLine.sigma);
        DoubleVertex B = new GaussianVertex(secondCsvLine.mu, secondCsvLine.sigma);

        //Noisily observe that the gaussian defined in the first line plus the gaussian in the
        //second line sums to 2.0
        DoubleVertex C = new GaussianVertex(A.plus(B), 1.0);
        C.observe(2.0);

        //Create a BayesNet object from your model
		BayesNetDoubleAsContinuous bayesNet = new BayesNetDoubleAsContinuous(A.getConnectedGraph());

        //Find the most probable value for A and B given we've taken a
        //noisy observation of 2.0
        GradientOptimizer optimizer = new GradientOptimizer(bayesNet);
        optimizer.maxAPosteriori(5000);

        //Expose model results
        results = (A.getValue() + B.getValue());

        System.out.println("A is most probably " + A.getValue());
        System.out.println("B is most probably " + B.getValue());
        System.out.println("Most probable actual value of the sum " + results);

    }

}
