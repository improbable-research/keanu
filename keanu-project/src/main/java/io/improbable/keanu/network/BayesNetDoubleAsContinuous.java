package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class BayesNetDoubleAsContinuous extends BayesianNetwork {

    //Lazy evaluated
    private List<Vertex<Double>> continuousLatentVertices;
    private List<Vertex> discreteLatentVertices;

    public BayesNetDoubleAsContinuous(Collection<? extends Vertex> vertices) {
        super(vertices);
    }

    public List<Vertex<Double>> getContinuousLatentVertices() {
        if (continuousLatentVertices == null) {
            splitContinuousAndDiscrete();
        }

        return continuousLatentVertices;
    }

    public List<Vertex> getDiscreteLatentVertices() {
        if (discreteLatentVertices == null) {
            splitContinuousAndDiscrete();
        }

        return discreteLatentVertices;
    }

    private void splitContinuousAndDiscrete() {

        continuousLatentVertices = new ArrayList<>();
        discreteLatentVertices = new ArrayList<>();

        for (Vertex<?> vertex : getLatentVertices()) {
            if (vertex.getValue() instanceof Double) {
                continuousLatentVertices.add((Vertex<Double>) vertex);
            } else {
                discreteLatentVertices.add(vertex);
            }
        }
    }

}
