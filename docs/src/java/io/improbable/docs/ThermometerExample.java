package io.improbable.docs;

//%%SNIPPET_START%% TempFull
import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class ThermometerExample {

    public static void main(String[] args) {

        //%%SNIPPET_START%% TempExpectedValues
        UniformVertex temperature = new UniformVertex(20., 30.);
        //%%SNIPPET_END%% TempExpectedValues

        //%%SNIPPET_START%% TempThermModel
        GaussianVertex firstThermometer = new GaussianVertex(temperature, 2.5);
        GaussianVertex secondThermometer = new GaussianVertex(temperature, 5.);
        //%%SNIPPET_END%% TempThermModel

        //%%SNIPPET_START%% TempObservations
        firstThermometer.observe(25.);
        secondThermometer.observe(30.);
        //%%SNIPPET_END%% TempObservations

        //%%SNIPPET_START%% TempMostProbable
        BayesianNetwork bayesNet = new BayesianNetwork(temperature.getConnectedGraph());
        Optimizer optimizer = Optimizer.of(bayesNet);
        optimizer.maxAPosteriori();

        double calculatedTemperature = temperature.getValue().scalar();
        //%%SNIPPET_END%% TempMostProbable

        System.out.println("Calculated Room Temperature: " + calculatedTemperature);
    }
}
//%%SNIPPET_END%% TempFull