package io.improbable.keanu.vertices.generic.probabilistic.discrete;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Collectors;

import io.improbable.keanu.distributions.discrete.Categorical;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class CategoricalVertex<T> extends Vertex<T> implements Probabilistic<T> {

    private final Map<T, DoubleVertex> selectableValues;

    public static <T> CategoricalVertex<T> of(Map<T, Double> selectableValues) {
        return new CategoricalVertex<>(defensiveCopy(selectableValues));
    }

    private static <T> Map<T, DoubleVertex> defensiveCopy(Map<T, Double> selectableValues) {
        LinkedHashMap<T, DoubleVertex> copy = new LinkedHashMap<>();
        for (Map.Entry<T, Double> entry : selectableValues.entrySet()) {
            copy.put(entry.getKey(), ConstantVertex.of(entry.getValue()));
        }
        return copy;
    }

    public CategoricalVertex(Map<T, DoubleVertex> selectableValues) {
        this.selectableValues = selectableValues;
        setParents(this.selectableValues.values());
    }

    public Map<T, DoubleVertex> getSelectableValues() {
        return selectableValues;
    }

    @Override
    public T sample(KeanuRandom random) {
        Categorical<T> categorical = Categorical.withParameters(selectableValuesMappedToDoubleTensor());
        return categorical.sample(getShape(), random);
    }

    @Override
    public double logProb(T value) {
        Categorical<T> categorical = Categorical.withParameters(selectableValuesMappedToDoubleTensor());
        return categorical.logProb(value).sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(T value) {
        return Collections.emptyMap();
    }

    private Map<T, DoubleTensor> selectableValuesMappedToDoubleTensor() {
        return selectableValues.entrySet()
            .stream()
            .collect(Collectors.toMap(Map.Entry::getKey, e -> e.getValue().getValue()));
    }
}
