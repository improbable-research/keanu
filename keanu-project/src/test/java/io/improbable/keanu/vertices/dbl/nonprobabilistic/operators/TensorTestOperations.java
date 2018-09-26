package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.number.IsCloseTo.closeTo;

import static io.improbable.keanu.tensor.TensorMatchers.allCloseTo;

import java.util.List;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class TensorTestOperations {

    public static void finiteDifferenceMatchesGradient(List<DoubleVertex> inputVertices,
                                                       DoubleVertex outputVertex,
                                                       final double incrementAmount,
                                                       final double delta,
                                                       final boolean doReverse) {

        finiteDifferenceMatchesForwardModeGradient(inputVertices, outputVertex, incrementAmount, delta);
        if (doReverse) {
            finiteDifferenceMatchesReverseModeGradient(inputVertices, outputVertex, incrementAmount, delta);
        }
    }

    public static void finiteDifferenceMatchesForwardModeGradient(List<DoubleVertex> inputVertices,
                                                                  DoubleVertex outputVertex,
                                                                  final double incrementAmount,
                                                                  final double delta) {
        inputVertices.forEach(v ->
            runGradientTestOnSingleInput(v, outputVertex, incrementAmount, delta, true));
    }

    public static void finiteDifferenceMatchesReverseModeGradient(List<DoubleVertex> inputVertices,
                                                                  DoubleVertex outputVertex,
                                                                  final double incrementAmount,
                                                                  final double delta) {
        inputVertices.forEach(v ->
            runGradientTestOnSingleInput(v, outputVertex, incrementAmount, delta, false));
    }

    private static void runGradientTestOnSingleInput(DoubleVertex inputVertex,
                                                     DoubleVertex outputVertex,
                                                     final double incrementAmount,
                                                     final Double delta,
                                                     final boolean forwardMode) {

        DoubleTensor initialInput = inputVertex.getValue();
        DoubleTensor initialOutput = outputVertex.eval();
        DoubleTensor outputWrtInput = dOutputWrtInput(outputVertex, inputVertex, forwardMode);

        long inputLength = TensorShape.getLength(inputVertex.getShape());
        int[] inputStride = TensorShape.getRowFirstStride(inputVertex.getShape());

        long outputLength = TensorShape.getLength(outputVertex.getShape());
        int[] outputStride = TensorShape.getRowFirstStride(outputVertex.getShape());

        for (int i = 0; i < inputLength; i++) {

            int[] inputIndex = TensorShape.getShapeIndices(inputVertex.getShape(), inputStride, i);
            DoubleTensor incrementTensor = DoubleTensor.zeros(inputVertex.getShape());
            incrementTensor.setValue(incrementAmount, inputIndex);

            inputVertex.setValue(initialInput.plus(incrementTensor));

            DoubleTensor newOutput = outputVertex.eval();
            DoubleTensor differenceInOutput = newOutput.minus(initialOutput);

            for (int j = 0; j < outputLength; j++) {

                int[] outputIndex = TensorShape.getShapeIndices(outputVertex.getShape(), outputStride, j);

                Double dOutputAtIndexWrtInputAtIndex = outputWrtInput.getValue(TensorShape.concat(outputIndex, inputIndex));
                Double diffAtOutputIndexByGradient = dOutputAtIndexWrtInputAtIndex * incrementAmount;
                Double diffAtOutputIndex = differenceInOutput.getValue(outputIndex);

                assertThat(diffAtOutputIndexByGradient, closeTo(diffAtOutputIndex, delta));
            }
        }
    }

    private static DoubleTensor dOutputWrtInput(DoubleVertex outputVertex, DoubleVertex inputVertex, boolean forwardMode) {

        if (forwardMode) {
            return outputVertex.getDualNumber().getPartialDerivatives().withRespectTo(inputVertex);
        } else {
            return Differentiator.reverseModeAutoDiff(outputVertex, inputVertex).withRespectTo(inputVertex);
        }
    }

}
