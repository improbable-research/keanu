package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators;

import static org.junit.Assert.assertThat;

import static io.improbable.keanu.tensor.TensorMatchers.allCloseTo;

import java.util.List;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class TensorTestOperations {
    public static void finiteDifferenceMatchesForwardAndReverseModeGradient(List<DoubleVertex> inputVertices,
                                                                            DoubleVertex outputVertex,
                                                                            double incrementAmount,
                                                                            Double delta) {
        finiteDifferenceMatchesForwardModeGradient(inputVertices, outputVertex, incrementAmount, delta);
        finiteDifferenceMatchesReverseModeGradient(inputVertices, outputVertex, incrementAmount, delta);
    }

    public static void finiteDifferenceMatchesForwardModeGradient(List<DoubleVertex> inputVertices,
                                                                  DoubleVertex outputVertex,
                                                                  double incrementAmount,
                                                                  double delta) {
        inputVertices.forEach(v ->
            runGradientTestOnSingleInput(v, outputVertex, incrementAmount, delta, true));
    }

    public static void finiteDifferenceMatchesReverseModeGradient(List<DoubleVertex> inputVertices,
                                                                  DoubleVertex outputVertex,
                                                                  double incrementAmount,
                                                                  double delta) {
        inputVertices.forEach(v ->
            runGradientTestOnSingleInput(v, outputVertex, incrementAmount, delta, false));
    }

    private static void runGradientTestOnSingleInput(DoubleVertex inputVertex,
                                                     DoubleVertex outputVertex,
                                                     double incrementAmount,
                                                     Double delta,
                                                     boolean isForwardMode) {
        DoubleTensor initialInput = inputVertex.getValue();

        DoubleTensor initialOutput = outputVertex.eval();
        DoubleTensor outputWrtInput = dOutputWrtInput(outputVertex, inputVertex, isForwardMode);

        int[] dimensionsToSumOver = getWrtDimensions(inputVertex, outputVertex);

        for (int i = 0; i < TensorShape.getLength(inputVertex.getShape()); i++) {
            DoubleTensor incrementTensor = DoubleTensor.zeros(inputVertex.getShape());
            incrementTensor.getFlattenedView().set(i, incrementAmount);
            inputVertex.setValue(initialInput.plus(incrementTensor));

            DoubleTensor newOutput = outputVertex.eval();
            DoubleTensor differenceInOutput = newOutput.minus(initialOutput);
            DoubleTensor differenceUsingGradient = outputWrtInput.times(incrementTensor).sum(dimensionsToSumOver);
            assertThat(gradientAssertMessage(outputVertex, inputVertex, isForwardMode),
                differenceUsingGradient, allCloseTo(delta, differenceInOutput));
        }
    }

    private static DoubleTensor dOutputWrtInput(DoubleVertex outputVertex, DoubleVertex inputVertex, boolean isForwardMode) {
        if (isForwardMode) {
            return outputVertex.getDualNumber().withRespectTo(inputVertex);
        } else {
            return Differentiator.reverseModeAutoDiff(outputVertex, inputVertex).withRespectTo(inputVertex);
        }
    }

    private static String gradientAssertMessage(DoubleVertex outputVertex, DoubleVertex inputVertex, boolean isForwardMode) {
        String descriptor = isForwardMode ? "Forward" : "Reverse";
        return String.format("%s derivative of vertex with ID %s with respect to %s should match finite difference", descriptor, outputVertex.getId(), inputVertex.getId());
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
