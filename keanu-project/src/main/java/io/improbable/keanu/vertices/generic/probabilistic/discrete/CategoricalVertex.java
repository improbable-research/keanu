package io.improbable.keanu.vertices.generic.probabilistic.discrete;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.discrete.Categorical;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.TakeVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.DirichletVertex;
import io.improbable.keanu.vertices.generic.GenericTensorVertex;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;
import static java.util.stream.Collectors.toMap;


public class CategoricalVertex<CATEGORY, TENSOR extends Tensor<CATEGORY>> extends GenericTensorVertex<TENSOR> implements Probabilistic<TENSOR>, NonSaveableVertex {

    private final Map<CATEGORY, DoubleVertex> selectableValues;

    public static <CATEGORY, TENSOR extends Tensor<CATEGORY>> CategoricalVertex<CATEGORY, TENSOR> of(Map<CATEGORY, Double> selectableValues) {
        return new CategoricalVertex<>(toDoubleVertices(selectableValues));
    }

    private static <CATEGORY> Map<CATEGORY, DoubleVertex> toDoubleVertices(Map<CATEGORY, Double> selectableValues) {
        return selectableValues.entrySet().stream()
            .collect(toMap(
                Map.Entry::getKey,
                e -> ConstantVertex.of(e.getValue())
                )
            );
    }

    public static <CATEGORY, TENSOR extends Tensor<CATEGORY>> CategoricalVertex<CATEGORY, TENSOR> of(DirichletVertex vertex, List<CATEGORY> categories) {

        final long length = TensorShape.getLength(vertex.getShape());
        if (length != categories.size()) {
            throw new IllegalArgumentException("Categories must have length of vertex's size");
        }

        final int categoriesCount = categories.size();
        final IntStream categoriesIndices = IntStream.range(0, categoriesCount);
        final Map<CATEGORY, DoubleVertex> selectableValues = categoriesIndices.boxed()
            .collect(
                toMap(
                    categories::get,
                    index -> new TakeVertex(vertex, index)
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

    public CategoricalVertex(long[] tensorShape, Map<CATEGORY, DoubleVertex> selectableValues) {
        super(tensorShape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(tensorShape, selectableValuesShapes(selectableValues));

        this.selectableValues = selectableValues;

        setParents(this.selectableValues.values());
    }

    public CategoricalVertex(Map<CATEGORY, DoubleVertex> selectableValues) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(selectableValuesShapes(selectableValues)), selectableValues);
    }

    public Map<CATEGORY, DoubleVertex> getSelectableValues() {
        return selectableValues;
    }

    @Override
    public TENSOR sample(KeanuRandom random) {
        Categorical<CATEGORY, TENSOR> categorical =
            Categorical.withParameters(selectableValuesMappedToDoubleTensor());
        return categorical.sample(getShape(), random);
    }

    @Override
    public double logProb(TENSOR value) {
        Categorical<CATEGORY, TENSOR> categorical = Categorical.
            withParameters(selectableValuesMappedToDoubleTensor());
        return categorical.logProb(value).sum();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(TENSOR value, Set<? extends Vertex> withRespectTo) {
        return Collections.emptyMap();
    }

    private Map<CATEGORY, DoubleTensor> selectableValuesMappedToDoubleTensor() {
        return selectableValues.entrySet().stream()
            .collect(toMap(
                Map.Entry::getKey,
                e -> e.getValue().getValue())
            );
    }

    private static long[][] selectableValuesShapes(Map<?, DoubleVertex> selectableValues) {
        return selectableValues.values().stream()
            .map(Vertex::getShape)
            .toArray(long[][]::new);
    }
}
