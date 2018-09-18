package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators;

import static io.improbable.keanu.tensor.TensorMatchers.allCloseTo;
import static org.hamcrest.MatcherAssert.assertThat;

import java.util.List;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class TensorTestOperations {

    public static void finiteDifferenceMatchesGradient(List<DoubleVertex> inputVertices,
                                                       DoubleVertex outputVertex,
                                                       final double INCREMENT_AMOUNT,
                                                       final double DELTA,
                                                       final boolean DO_REVERSE) {
        setInitialConditions(inputVertices);
        inputVertices.forEach(v ->
            runGradientTestOnSingleInput(v, outputVertex, INCREMENT_AMOUNT, DELTA, DO_REVERSE));
    }

    private static void setInitialConditions(List<DoubleVertex> inputVertices) {
        inputVertices.forEach(v -> v.setValue(v.sample()));
    }

    private static void runGradientTestOnSingleInput(DoubleVertex inputVertex,
                                                     DoubleVertex outputVertex,
                                                     final double INCREMENT_AMOUNT,
                                                     final double DELTA,
                                                     final boolean DO_REVERSE) {
        DoubleTensor initialInput = inputVertex.getValue();

        Double boxedDelta = DELTA;
        DoubleTensor initialOutput = outputVertex.eval();
        DoubleTensor outputWrtInput = outputVertex.getDualNumber().getPartialDerivatives().withRespectTo(inputVertex);

        if (DO_REVERSE) {
            DoubleTensor reverseDifferential =
                Differentiator.reverseModeAutoDiff(outputVertex, inputVertex).withRespectTo(inputVertex);

            assertThat(reverseDifferential, allCloseTo(boxedDelta, outputWrtInput));
        }

        int[] dimensionsToSumOver = getWrtDimensions(inputVertex, outputVertex);
        DoubleTensor incrementTensor = DoubleTensor.zeros(inputVertex.getShape());
        Tensor.FlattenedView<Double> flatIncrement = incrementTensor.getFlattenedView();

        for (int i = 0; i < flatIncrement.size(); i++) {
            flatIncrement.set(i, INCREMENT_AMOUNT);
            inputVertex.setValue(initialInput.plus(incrementTensor));

            DoubleTensor newOutput = outputVertex.eval();
            DoubleTensor differenceInOutput = newOutput.minus(initialOutput);
            DoubleTensor differenceUsingGradient = outputWrtInput.times(incrementTensor).sum(dimensionsToSumOver);
            assertThat(differenceUsingGradient, allCloseTo(boxedDelta, differenceInOutput));
        }
    }

    private static int[] getWrtDimensions(DoubleVertex wrtVertex,
                                          DoubleVertex ofVertex) {
        int wrtRank = wrtVertex.getShape().length;
        int ofRank = ofVertex.getShape().length;
        int[] wrtDimensions = new int[wrtRank];

        for (int i = 0; i < wrtRank; i++) {
            wrtDimensions[i] = ofRank + i;
        }

        return wrtDimensions;
    }

}
