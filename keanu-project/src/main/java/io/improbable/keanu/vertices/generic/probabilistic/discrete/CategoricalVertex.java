package io.improbable.keanu.vertices.generic.probabilistic.discrete;

import io.improbable.keanu.distributions.discrete.Categorical;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.TakeVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.DirichletVertex;
import org.nd4j.linalg.util.ArrayUtil;

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
        final int length = ArrayUtil.prod(vertex.getShape());
        if (length != categories.size()) {
            throw new IllegalArgumentException("Categories must have length of vertex's size");
        }

        final int categoriesCount = categories.size();

        return new CategoricalVertex<>(IntStream.range(0, categoriesCount).boxed().collect(toMap(categories::get, i -> new TakeVertex(vertex, 0, i))));
    }

    public static CategoricalVertex<Integer> of(DirichletVertex vertex) {
        final int categoriseCount = ArrayUtil.prod(vertex.getShape());
        return CategoricalVertex.of(vertex, IntStream.range(0, categoriseCount).boxed().collect(Collectors.toList()));
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
    public Map<Vertex, DoubleTensor> dLogProb(T value, Set<? extends Vertex> withRespectTo) {
        return Collections.emptyMap();
    }

    private Map<T, DoubleTensor> selectableValuesMappedToDoubleTensor() {
        return selectableValues.entrySet().stream()
            .collect(toMap(Map.Entry::getKey, e -> e.getValue().getValue()));
    }
}
