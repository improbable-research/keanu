package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkShapesCanBeConcatenated;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.apache.commons.math3.util.Pair;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class ConcatenationVertex extends DoubleVertex implements Differentiable, NonProbabilistic<DoubleTensor> {

    private final int dimension;
    private final DoubleVertex[] input;

    /**
     * A vertex that can concatenate any amount of vertices along a given dimension.
     *
     * @param dimension the dimension to concatenate on. This is the only dimension in which sizes may be different.
     * @param input     the input vertices to concatenate
     */
    public ConcatenationVertex(int dimension, DoubleVertex... input) {
        this.dimension = dimension;
        this.input = input;
        setParents(input);
        int[][] shapes = extractFromInputs(int[].class, Vertex::getShape);
        setValue(DoubleTensor.placeHolder(checkShapesCanBeConcatenated(dimension, shapes)));
    }

    @Override
    public PartialDerivatives forwardModeAutoDifferentiation(Map<Vertex, PartialDerivatives> derivativeOfParentsWithRespectToInputs) {
        List<PartialDerivatives> patialsOfInputs = new ArrayList<>();

        for (DoubleVertex vertex : input) {
            patialsOfInputs.add(derivativeOfParentsWithRespectToInputs.get(vertex));
        }

        return concat(derivativeOfParentsWithRespectToInputs, patialsOfInputs, input, dimension);
    }

    public static PartialDerivatives concat(Map<Vertex, PartialDerivatives> partials,
                                            List<PartialDerivatives> partialsOfInputs,
                                            DoubleVertex[] input,
                                            int dimension) {

        Map<VertexId, List<DoubleTensor>> partialsToConcat = new HashMap<>();
        List<Pair<VertexId, List<Integer>>> vertexIds = findShapesOfVertexWithRespectTo(partialsOfInputs);

        for (Pair<VertexId, List<Integer>> wrtVertex : vertexIds) {
            VertexId vertexId = wrtVertex.getFirst();

            for (DoubleVertex ofVertex : input) {
                int[] shapeOfVertexWithRespectTo = wrtVertex.getSecond().stream().mapToInt(i -> i).toArray();
                int[] wrtVertexShape = Arrays.copyOfRange(shapeOfVertexWithRespectTo, ofVertex.getValue().getRank(), wrtVertex.getSecond().size());
                int[] shape = TensorShape.concat(ofVertex.getShape(), wrtVertexShape);
                PartialDerivatives partialOf = partials.get(ofVertex);

                if (partialOf.asMap().containsKey(vertexId)) {
                    partialsToConcat.computeIfAbsent(vertexId, k -> new ArrayList<>()).add(partialOf.asMap().get(vertexId));
                } else {
                    partialsToConcat.computeIfAbsent(vertexId, k -> new ArrayList<>()).add(DoubleTensor.zeros(shape));
                }

            }
        }

        Map<VertexId, DoubleTensor> concattedPartials = new HashMap<>();

        for (Map.Entry<VertexId, List<DoubleTensor>> partialForVertex : partialsToConcat.entrySet()) {
            DoubleTensor concatted = concatPartialDerivates(dimension, partialForVertex.getValue());
            concattedPartials.put(partialForVertex.getKey(), concatted);
        }

        return new PartialDerivatives(concattedPartials);
    }

    private static DoubleTensor concatPartialDerivates(int dimension, List<DoubleTensor> partialDerivates) {
        if (partialDerivates.size() == 1) {
            return partialDerivates.get(0);
        } else {
            DoubleTensor[] derivativesToConcat = new DoubleTensor[partialDerivates.size()];
            return DoubleTensor.concat(dimension, partialDerivates.toArray(derivativesToConcat));
        }
    }

    private static List<Pair<VertexId, List<Integer>>> findShapesOfVertexWithRespectTo(List<PartialDerivatives> partials) {
        List<Pair<VertexId, List<Integer>>> vertexInfo = new ArrayList<>();
        Set ids = new HashSet();

        for (PartialDerivatives partial : partials) {
            for (Map.Entry<VertexId, DoubleTensor> entry : partial.asMap().entrySet()) {
                if (!ids.contains(entry.getKey())) {
                    vertexInfo.add(new Pair<>(entry.getKey(), Arrays.stream(entry.getValue().getShape()).boxed().collect(Collectors.toList())));
                    ids.add(entry.getKey());
                }
            }
        }

        return vertexInfo;
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> splitPartials = new HashMap<>();

        int currentSplitIndex = 0;
        int[] splitIndices = new int[input.length];

        for (int i = 0; i < input.length; i++) {
            splitIndices[i] = currentSplitIndex + input[i].getShape()[dimension];
            currentSplitIndex = splitIndices[i];
            splitPartials.put(input[i], new PartialDerivatives(new HashMap<>()));
        }

        int wrtDimensionToSliceOn = input[0].getShape().length + dimension;
        for (Map.Entry<VertexId, DoubleTensor> entry : derivativeOfOutputsWithRespectToSelf.asMap().entrySet()) {
            DoubleTensor partial = entry.getValue();

            List<DoubleTensor> splitPartial = partial.split(wrtDimensionToSliceOn, splitIndices);

            for (int i = 0; i < splitPartial.size(); i++) {
                splitPartials.get(input[i]).putWithRespectTo(entry.getKey(), splitPartial.get(i));
            }

        }

        return splitPartials;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op(extractFromInputs(DoubleTensor.class, Vertex::sample));
    }

    @Override
    public DoubleTensor calculate() {
        return op(extractFromInputs(DoubleTensor.class, Vertex::getValue));
    }

    protected DoubleTensor op(DoubleTensor... inputs) {
        return DoubleTensor.concat(dimension, inputs);
    }

    private <T> T[] extractFromInputs(Class<T> clazz, Function<Vertex<DoubleTensor>, T> func) {
        T[] extract = (T[]) Array.newInstance(clazz, input.length);
        for (int i = 0; i < input.length; i++) {
            extract[i] = func.apply(input[i]);
        }
        return extract;
    }

}
