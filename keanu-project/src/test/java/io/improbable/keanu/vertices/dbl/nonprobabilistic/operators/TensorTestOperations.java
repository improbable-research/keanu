package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators;

import static io.improbable.keanu.tensor.TensorMatchers.allCloseTo;
import static org.hamcrest.MatcherAssert.assertThat;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import java.util.List;

public class TensorTestOperations {

    public static void finiteDifferenceMatchesGradient(
            List<DoubleVertex> inputVertices,
            DoubleVertex outputVertex,
            final double incrementAmount,
            final Double delta,
            final boolean doReverse) {
        inputVertices.forEach(
                v ->
                        runGradientTestOnSingleInput(
                                v, outputVertex, incrementAmount, delta, doReverse));
    }

    private static void runGradientTestOnSingleInput(
            DoubleVertex inputVertex,
            DoubleVertex outputVertex,
            final double incrementAmount,
            final Double delta,
            final boolean doReverse) {
        DoubleTensor initialInput = inputVertex.getValue();

        DoubleTensor initialOutput = outputVertex.eval();
        DoubleTensor outputWrtInputForward =
                outputVertex.getDualNumber().getPartialDerivatives().withRespectTo(inputVertex);

        if (doReverse) {
            DoubleTensor outputWrtInputReverse =
                    Differentiator.reverseModeAutoDiff(outputVertex, inputVertex)
                            .withRespectTo(inputVertex);
            assertThat(outputWrtInputForward, allCloseTo(delta, outputWrtInputReverse));
        }

        int[] dimensionsToSumOver = getWrtDimensions(inputVertex, outputVertex);

        for (int i = 0; i < TensorShape.getLength(inputVertex.getShape()); i++) {
            DoubleTensor incrementTensor = DoubleTensor.zeros(inputVertex.getShape());
            incrementTensor.getFlattenedView().set(i, incrementAmount);
            inputVertex.setValue(initialInput.plus(incrementTensor));

            DoubleTensor newOutput = outputVertex.eval();
            DoubleTensor differenceInOutput = newOutput.minus(initialOutput);
            DoubleTensor differenceUsingGradient =
                    outputWrtInputForward.times(incrementTensor).sum(dimensionsToSumOver);
            assertThat(differenceUsingGradient, allCloseTo(delta, differenceInOutput));
        }
    }

    private static int[] getWrtDimensions(DoubleVertex wrtVertex, DoubleVertex ofVertex) {
        int wrtRank = wrtVertex.getShape().length;
        int ofRank = ofVertex.getShape().length;
        int[] wrtDimensions = new int[wrtRank];

        for (int i = 0; i < wrtRank; i++) {
            wrtDimensions[i] = ofRank + i;
        }

        return wrtDimensions;
    }
}
