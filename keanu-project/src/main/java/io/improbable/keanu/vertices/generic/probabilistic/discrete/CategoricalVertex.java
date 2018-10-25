package io.improbable.keanu.vertices.generic.probabilistic.discrete;

import io.improbable.keanu.distributions.discrete.Categorical;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.TakeVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.DirichletVertex;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;
import static java.util.stream.Collectors.toMap;


public class CategoricalVertex<T, TENSOR extends Tensor<T>> extends Vertex<TENSOR> implements Probabilistic<TENSOR> {

    private final Map<T, DoubleVertex> selectableValues;

    public static <T, TENSOR extends Tensor<T>> CategoricalVertex<T, TENSOR> of(Map<T, Double> selectableValues) {
        return new CategoricalVertex<>(toDoubleVertices(selectableValues));
    }

    private static <T> Map<T, DoubleVertex> toDoubleVertices(Map<T, Double> selectableValues) {
        Map<T, DoubleVertex> copy = new LinkedHashMap<>();
        for (Map.Entry<T, Double> entry : selectableValues.entrySet()) {
            copy.put(entry.getKey(), ConstantVertex.of(entry.getValue()));
        }
        return copy;
    }

    public static <T, TENSOR extends Tensor<T>> CategoricalVertex<T, TENSOR> of(DirichletVertex vertex, List<T> categories) {

        final long length = TensorShape.getLength(vertex.getShape());
        if (length != categories.size()) {
            throw new IllegalArgumentException("Categories must have length of vertex's size");
        }

        final int categoriesCount = categories.size();
        final IntStream categoriesIndices = IntStream.range(0, categoriesCount);
        final Map<T, DoubleVertex> selectableValues = categoriesIndices.boxed()
            .collect(
                toMap(
                    categories::get,
                    index -> new TakeVertex(vertex, 0, index)
                )
            );
        return new CategoricalVertex<>(selectableValues);
    }

    public static CategoricalVertex<Integer, IntegerTensor> of(DirichletVertex vertex) {
        final int categoriesCount = Math.toIntExact(TensorShape.getLength(vertex.getShape()));
        final IntStream categories = IntStream.range(0, categoriesCount);
        final List<Integer> categoriesList = categories.boxed().collect(Collectors.toList());
        return CategoricalVertex.of(vertex, categoriesList);
    }

    public CategoricalVertex(long[] tensorShape, Map<T, DoubleVertex> selectableValues) {
        super(tensorShape);
        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, selectableValuesShapes(selectableValues));

        this.selectableValues = selectableValues;

        setParents(this.selectableValues.values());
    }

    public CategoricalVertex(Map<T, DoubleVertex> selectableValues) {
        this(checkHasSingleNonScalarShapeOrAllScalar(selectableValuesShapes(selectableValues)), selectableValues);
    }

    public Map<T, DoubleVertex> getSelectableValues() {
        return selectableValues;
    }

    @Override
    public TENSOR sample(KeanuRandom random) {
        Categorical<T, TENSOR> categorical =
            Categorical.withParameters(selectableValuesMappedToDoubleTensor());
        return categorical.sample(getShape(), random);
    }

    @Override
    public double logProb(TENSOR value) {
        Categorical<T, TENSOR> categorical = Categorical.
            withParameters(selectableValuesMappedToDoubleTensor());
        return categorical.logProb(value).sum();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(TENSOR value, Set<? extends Vertex> withRespectTo) {
        return Collections.emptyMap();
    }

    private Map<T, DoubleTensor> selectableValuesMappedToDoubleTensor() {
        return selectableValues.entrySet().stream().collect(toMap(Map.Entry::getKey,
            e -> e.getValue().getValue()));
    }

    private static long[][] selectableValuesShapes(Map<?, DoubleVertex> selectableValues) {
        return selectableValues.values().stream().map(Vertex::getShape).toArray(long[][]::new);
    }
}
