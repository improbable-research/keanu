package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import java.util.List;

import static io.improbable.keanu.tensor.TensorMatchers.allCloseTo;
import static org.junit.Assert.assertThat;

public class TensorTestOperations {
    public static <T extends DoubleVertex & Differentiable>
    void finiteDifferenceMatchesForwardAndReverseModeGradient(List<T> inputVertices,
                                                              T outputVertex,
                                                              double incrementAmount,
                                                              Double delta) {
        finiteDifferenceMatchesForwardModeGradient(inputVertices, outputVertex, incrementAmount, delta);
        finiteDifferenceMatchesReverseModeGradient(inputVertices, outputVertex, incrementAmount, delta);
    }

    public static <T extends DoubleVertex & Differentiable>
    void finiteDifferenceMatchesForwardModeGradient(List<T> inputVertices,
                                                    T outputVertex,
                                                    double incrementAmount,
                                                    double delta) {
        inputVertices.forEach(v ->
            runGradientTestOnSingleInput(v, outputVertex, incrementAmount, delta, true));
    }

    public static <T extends DoubleVertex & Differentiable>
    void finiteDifferenceMatchesReverseModeGradient(List<T> inputVertices,
                                                    T outputVertex,
                                                    double incrementAmount,
                                                    double delta) {
        inputVertices.forEach(v ->
            runGradientTestOnSingleInput(v, outputVertex, incrementAmount, delta, false));
    }

    private static <T extends DoubleVertex & Differentiable>
    void runGradientTestOnSingleInput(T inputVertex,
                                      T outputVertex,
                                      double incrementAmount,
                                      double delta,
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

    private static <T extends DoubleVertex & Differentiable>
    DoubleTensor dOutputWrtInput(T outputVertex, T inputVertex, boolean isForwardMode) {
        if (isForwardMode) {
            return Differentiator.forwardModeAutoDiff(inputVertex, outputVertex).of(outputVertex);
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
