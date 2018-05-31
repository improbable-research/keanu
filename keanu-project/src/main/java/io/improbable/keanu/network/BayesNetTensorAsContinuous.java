package io.improbable.keanu.network;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class BayesNetTensorAsContinuous extends BayesianNetwork {

    //Lazy evaluated
    private List<Vertex<DoubleTensor>> continuousLatentVertices;
    private List<Vertex> discreteLatentVertices;

    public BayesNetTensorAsContinuous(Collection<? extends Vertex> vertices) {
        super(vertices);
    }

    public List<Vertex<DoubleTensor>> getContinuousLatentVertices() {
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
            if (vertex.getValue() instanceof DoubleTensor) {
                continuousLatentVertices.add((Vertex<DoubleTensor>) vertex);
            } else {
                discreteLatentVertices.add(vertex);
            }
        }
    }

}
