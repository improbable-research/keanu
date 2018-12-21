package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

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
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        List<PartialDerivative> partialsOfOperands = new ArrayList<>();
        List<DoubleTensor> operandValues = new ArrayList<>();

        for (DoubleVertex operand : operands) {
            PartialDerivative operandPartial = derivativeOfParentsWithRespectToInput.getOrDefault(operand, PartialDerivative.EMPTY);
            partialsOfOperands.add(operandPartial);
            operandValues.add(operand.getValue());
        }

        return concat(partialsOfOperands, operandValues, dimension);
    }

    public static PartialDerivative concat(List<PartialDerivative> partialsOfOperands,
                                           List<DoubleTensor> operandValues,
                                           int dimension) {

        long[] wrtShape = null;
        for (int i = 0; i < partialsOfOperands.size(); i++) {
            PartialDerivative partial = partialsOfOperands.get(i);
            DoubleTensor operandValue = operandValues.get(i);

            if (partial.isPresent()) {
                long[] partialWrtShape = partial.get().getShape();
                wrtShape = Arrays.copyOfRange(partialWrtShape, operandValue.getRank(), partialWrtShape.length);
                break;
            }
        }

        List<DoubleTensor> partialsToConcat = getPartialsToConcatForInput(
            partialsOfOperands,
            operandValues,
            wrtShape
        );

        return new PartialDerivative(concatPartialDerivatives(dimension, partialsToConcat));
    }

    private static List<DoubleTensor> getPartialsToConcatForInput(List<PartialDerivative> partialsOfOperands,
                                                                  List<DoubleTensor> operandValues,
                                                                  long[] wrtShape) {

        List<DoubleTensor> partialsToConcat = new ArrayList<>();

        for (int i = 0; i < operandValues.size(); i++) {
            PartialDerivative partialOfOperand = partialsOfOperands.get(i);
            DoubleTensor operandValue = operandValues.get(i);

            if (partialOfOperand.isPresent()) {
                partialsToConcat.add(partialOfOperand.get());
            } else {
                long[] resultShape = TensorShape.concat(operandValue.getShape(), wrtShape);
                partialsToConcat.add(DoubleTensor.zeros(resultShape));
            }

        }

        return partialsToConcat;
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
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> splitPartials = new HashMap<>();

        long currentSplitIndex = 0;
        long[] splitIndices = new long[operands.length];

        for (int i = 0; i < operands.length; i++) {
            splitIndices[i] = currentSplitIndex + operands[i].getShape()[dimension];
            currentSplitIndex = splitIndices[i];
            splitPartials.put(operands[i], PartialDerivative.EMPTY);
        }

        int operandsRank = operands[0].getShape().length;
        int wrtStartsAt = -operandsRank;
        int wrtSplitOn = wrtStartsAt + dimension;

        DoubleTensor partial = derivativeOfOutputWithRespectToSelf.get();

        List<DoubleTensor> splitPartial = partial.split(wrtSplitOn, splitIndices);

        for (int i = 0; i < splitPartial.size(); i++) {
            splitPartials.put(operands[i], new PartialDerivative(splitPartial.get(i)));
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
