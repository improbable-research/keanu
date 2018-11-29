package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkShapesCanBeConcatenated;

public class ConcatenationVertex extends DoubleVertex implements Differentiable, NonProbabilistic<DoubleTensor> {

    private final static String DIMENSION_NAME = "dimension";
    private final static String OPERANDS_NAME = "operands";

    private final int dimension;
    private final DoubleVertex[] operands;

    /**
     * A vertex that can concatenate any amount of vertices along a given dimension.
     *
     * @param dimension the dimension to concatenate on. This is the only dimension in which sizes may be different. Negative
     *                  dimension indexing is not supported.
     * @param operands  the operands vertices to concatenate
     */
    public ConcatenationVertex(int dimension, DoubleVertex... operands) {
        super(checkShapesCanBeConcatenated(dimension, extractFromInputs(long[].class, Vertex::getShape, operands)));
        this.dimension = dimension;
        this.operands = operands;
        setParents(operands);
    }

    public ConcatenationVertex(@LoadVertexParam(DIMENSION_NAME) int dimension,
                               @LoadVertexParam(OPERANDS_NAME) Vertex[] operands) {
        this(dimension, convertFromVertexToDoubleVertex(operands));
    }

    private static DoubleVertex[] convertFromVertexToDoubleVertex(Vertex[] operands) {
        return Arrays.stream(operands).toArray(DoubleVertex[]::new);
    }

    @Override
    public PartialDerivatives forwardModeAutoDifferentiation(Map<Vertex, PartialDerivatives> derivativeOfParentsWithRespectToInputs) {
        List<PartialDerivatives> partialsOfInputs = new ArrayList<>();
        List<DoubleTensor> inputValues = new ArrayList<>();

        for (DoubleVertex operand : operands) {
            partialsOfInputs.add(derivativeOfParentsWithRespectToInputs.get(operand));
            inputValues.add(operand.getValue());
        }

        return concat(partialsOfInputs, inputValues, dimension);
    }

    public static PartialDerivatives concat(List<PartialDerivatives> derivativeOfOperandsWrtInputs,
                                            List<DoubleTensor> operandValues,
                                            int dimension) {

        Map<VertexId, List<DoubleTensor>> partialsToConcat = new HashMap<>();
        Map<VertexId, long[]> partialShapes = findShapesOfVertexWithRespectTo(derivativeOfOperandsWrtInputs);

        for (Map.Entry<VertexId, long[]> partialShape : partialShapes.entrySet()) {
            VertexId wrtVertexId = partialShape.getKey();
            long[] partialWrtShape = partialShape.getValue();

            getPartialsToConcatForInput(partialsToConcat, derivativeOfOperandsWrtInputs, operandValues, wrtVertexId, partialWrtShape);
        }

        return concatAll(partialsToConcat, dimension);
    }

    private static Map<VertexId, long[]> findShapesOfVertexWithRespectTo(List<PartialDerivatives> derivativeOfOperandsWrtInputs) {
        Map<VertexId, long[]> vertexInfo = new HashMap<>();

        for (PartialDerivatives derivativeOfOperandWrtInputs : derivativeOfOperandsWrtInputs) {

            for (Map.Entry<VertexId, DoubleTensor> entry : derivativeOfOperandWrtInputs.asMap().entrySet()) {
                vertexInfo.computeIfAbsent(entry.getKey(), (wrtId) -> entry.getValue().getShape());
            }
        }

        return vertexInfo;
    }

    private static void getPartialsToConcatForInput(Map<VertexId, List<DoubleTensor>> partialsToConcat,
                                                    List<PartialDerivatives> derivativeOfOperandsWrtInputs,
                                                    List<DoubleTensor> operandValues,
                                                    VertexId wrtVertexId,
                                                    long[] partialWrtShape) {

        for (int i = 0; i < operandValues.size(); i++) {
            PartialDerivatives partialOfOperand = derivativeOfOperandsWrtInputs.get(i);
            DoubleTensor operandValue = operandValues.get(i);

            if (partialOfOperand.asMap().containsKey(wrtVertexId)) {
                partialsToConcat.computeIfAbsent(wrtVertexId, k -> new ArrayList<>()).add(partialOfOperand.asMap().get(wrtVertexId));
            } else {
                long[] wrtShape = Arrays.copyOfRange(partialWrtShape, operandValue.getRank(), partialWrtShape.length);
                long[] resultShape = TensorShape.concat(operandValue.getShape(), wrtShape);

                partialsToConcat.computeIfAbsent(wrtVertexId, k -> new ArrayList<>()).add(DoubleTensor.zeros(resultShape));
            }

        }
    }

    private static PartialDerivatives concatAll(Map<VertexId, List<DoubleTensor>> partialsToConcat,
                                                int dimension) {

        Map<VertexId, DoubleTensor> concattedPartials = new HashMap<>();

        for (Map.Entry<VertexId, List<DoubleTensor>> partialForVertex : partialsToConcat.entrySet()) {
            DoubleTensor concatted = concatPartialDerivatives(dimension, partialForVertex.getValue());
            concattedPartials.put(partialForVertex.getKey(), concatted);
        }

        return new PartialDerivatives(concattedPartials);
    }

    private static DoubleTensor concatPartialDerivatives(int dimension, List<DoubleTensor> partialDerivatives) {
        if (partialDerivatives.size() == 1) {
            return partialDerivatives.get(0);
        } else {
            DoubleTensor[] derivativesToConcat = new DoubleTensor[partialDerivatives.size()];
            return DoubleTensor.concat(dimension, partialDerivatives.toArray(derivativesToConcat));
        }
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> splitPartials = new HashMap<>();

        long currentSplitIndex = 0;
        long[] splitIndices = new long[operands.length];

        for (int i = 0; i < operands.length; i++) {
            splitIndices[i] = currentSplitIndex + operands[i].getShape()[dimension];
            currentSplitIndex = splitIndices[i];
            splitPartials.put(operands[i], new PartialDerivatives(new HashMap<>()));
        }

        int operandsRank = operands[0].getShape().length;
        int wrtStartsAt = -operandsRank;
        int wrtSplitOn = wrtStartsAt + dimension;

        for (Map.Entry<VertexId, DoubleTensor> entry : derivativeOfOutputsWithRespectToSelf.asMap().entrySet()) {
            DoubleTensor partial = entry.getValue();

            List<DoubleTensor> splitPartial = partial.split(wrtSplitOn, splitIndices);

            for (int i = 0; i < splitPartial.size(); i++) {
                splitPartials.get(operands[i]).putWithRespectTo(entry.getKey(), splitPartial.get(i));
            }

        }

        return splitPartials;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op(extractFromInputs(DoubleTensor.class, Vertex::sample, operands));
    }

    @Override
    public DoubleTensor calculate() {
        return op(extractFromInputs(DoubleTensor.class, Vertex::getValue, operands));
    }

    protected DoubleTensor op(DoubleTensor... inputs) {
        return DoubleTensor.concat(dimension, inputs);
    }

    private static <T> T[] extractFromInputs(Class<T> clazz, Function<Vertex<DoubleTensor>, T> func, DoubleVertex[] operands) {
        T[] extract = (T[]) Array.newInstance(clazz, operands.length);
        for (int i = 0; i < operands.length; i++) {
            extract[i] = func.apply(operands[i]);
        }
        return extract;
    }

    @SaveVertexParam(OPERANDS_NAME)
    public DoubleVertex[] getOperands() {
        return operands;
    }

    @SaveVertexParam(DIMENSION_NAME)
    public int getDimension() {
        return dimension;
    }
}
