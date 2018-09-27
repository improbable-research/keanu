package io.improbable.keanu.vertices.generic.probabilistic.discrete;

import io.improbable.keanu.distributions.discrete.Categorical;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.TakeVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.DirichletVertex;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.toMap;

public class CategoricalVertex<T> extends Vertex<T> implements Probabilistic<T> {

    private final Map<T, DoubleVertex> selectableValues;

    public static <T> CategoricalVertex<T> of(Map<T, Double> selectableValues) {
        return new CategoricalVertex<>(defensiveCopy(selectableValues));
    }

    public static <T> CategoricalVertex<T> of(DirichletVertex vertex, List<T> categories) {
        final long length = TensorShape.getLength(vertex.getShape());
        if (length != categories.size()) {
            throw new IllegalArgumentException("Categories must have length of vertex's size");
        }

        final int categoriesCount = categories.size();
        final IntStream categoriesIndices = IntStream.range(0, categoriesCount);
        final Map<T, DoubleVertex> selectableValues =
            categoriesIndices.boxed().collect(toMap(categories::get,
                index -> new TakeVertex(vertex, 0, index)));
        return new CategoricalVertex<>(selectableValues);
    }

    public static CategoricalVertex<Integer> of(DirichletVertex vertex) {
        final long categoriesCountLong = TensorShape.getLength(vertex.getShape());

        // This should never happen in practice, but the number of categories must be an int
        // as they get stored in a Collection, which can only have the size of an int.
        if (categoriesCountLong > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Number of categories cannot exceed size of an integer");
        }

        // We cast to an int so we can expose a more predictable key type (Integer)
        final int categoriesCount = (int) categoriesCountLong;
        final IntStream categories = IntStream.range(0, categoriesCount);
        final List<Integer> categoriesList = categories.boxed().collect(Collectors.toList());
        return CategoricalVertex.of(vertex, categoriesList);
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
        Categorical<T> categorical =
            Categorical.withParameters(selectableValuesMappedToDoubleTensor());
        return categorical.sample(getShape(), random);
    }

    @Override
    public double logProb(T value) {
        Categorical<T> categorical =
            Categorical.withParameters(selectableValuesMappedToDoubleTensor());
        return categorical.logProb(value).sum();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(T value, Set<? extends Vertex> withRespectTo) {
        return Collections.emptyMap();
    }

    private Map<T, DoubleTensor> selectableValuesMappedToDoubleTensor() {
        return selectableValues.entrySet().stream().collect(toMap(Map.Entry::getKey,
            e -> e.getValue().getValue()));
    }
}
