package com.example.starter;


import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.util.Random;

public class Model {

    public static void main(String[] args) {

        //Load data from a csv file
        Data data = Data.load("data_example.csv");

        //create my model using the data
        Model model = new Model(data);
        model.run();
    }

    final Data data;
    NetworkSamples results;

    public Model(Data data) {
        this.data = data;
    }

    public void run() {

        // create a random and set its seed if you want your model to run the same each time
        Random random = new Random(1);

        //Create your model as a bayesian network
        DoubleVertex A = new GaussianVertex(0, 1, random);
        DoubleVertex B = new GaussianVertex(A, 1, random);

        //Add your observations
        B.observe(0.5);

        //Create a BayesNet object from your model
        BayesNet bayesNet = new BayesNet(A.getConnectedGraph());

        //Run an inference algorithm on your bayesian
        NetworkSamples posteriorDistSamples = MetropolisHastings.getPosteriorSamples(
                bayesNet,
                //Specify which latent variables you're interested in. Probably all of them.
                bayesNet.getLatentVertices(),
                50000,
                random);

        //Drop samples and subsample to account for bias in the inference algorithm
        results = posteriorDistSamples.drop(1000).downSample(5);
    }

}
