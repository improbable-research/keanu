package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class TensorSmoothUniform {

    private TensorSmoothUniform() {
    }

    public static DoubleTensor sample(int[] shape, DoubleTensor xMin, DoubleTensor xMax, DoubleTensor edgeSharpness, KeanuRandom random) {

        final DoubleTensor r1 = random.nextDouble(shape);
        final DoubleTensor bodyWidth = xMax.minus(xMin);
        final DoubleTensor shoulderWidth = edgeSharpness.times(bodyWidth);
        final DoubleTensor rScaled = r1.times(bodyWidth.plus(shoulderWidth)).plus(xMin.minus(shoulderWidth.div(2)));

        final DoubleTensor firstConditional = rScaled.getGreaterThanOrEqualToMask(xMin);
        firstConditional.timesInPlace(rScaled.getLessThanOrEqualToMask(xMax));

        final DoubleTensor bodyHeight = bodyHeight(shoulderWidth, bodyWidth);
        final DoubleTensor r2 = random.nextDouble(shape);

        final DoubleTensor secondConditional = rScaled.getLessThanMask(xMin);
        final DoubleTensor spillOnToShoulder = xMin.minus(rScaled);
        final DoubleTensor shoulderX = shoulderWidth.minus(spillOnToShoulder);
        final DoubleTensor shoulderDensity = shoulder(shoulderWidth, bodyWidth, shoulderX);
        final DoubleTensor acceptProbability = shoulderDensity.div(bodyHeight);

        final DoubleTensor secondConditionalNestedTrue = secondConditional.times(r2.getLessThanOrEqualToMask(acceptProbability));
        final DoubleTensor secondConditionalNestedFalse = secondConditional.times(r2.getGreaterThanMask(acceptProbability));
        final DoubleTensor secondConditionalNestedFalseResult = xMin.minus(shoulderWidth).plus(spillOnToShoulder);

        final DoubleTensor secondConditionalFalse = rScaled.getGreaterThanOrEqualToMask(xMin);
        final DoubleTensor spillOnToShoulder2 = rScaled.minus(xMax);
        final DoubleTensor shoulderX2 = shoulderWidth.minus(spillOnToShoulder);
        final DoubleTensor shoulderDensity2 = shoulder(shoulderWidth, bodyWidth, shoulderX2);
        final DoubleTensor acceptProbability2 = shoulderDensity2.div(bodyHeight);

        final DoubleTensor secondConditionalFalseNestedTrue = secondConditionalFalse.times(r2.getLessThanOrEqualToMask(acceptProbability2));
        final DoubleTensor secondConditionalFalseNestedFalse = secondConditionalNestedFalse.times(r2.getGreaterThanMask(acceptProbability2));
        final DoubleTensor secondConditionalFalseNestedFalseResult = xMax.plus(shoulderWidth).minus(spillOnToShoulder2);

        return firstConditional.times(rScaled).
            plus(secondConditionalNestedTrue.times(rScaled)).
            plus(secondConditionalNestedFalse.times(secondConditionalNestedFalseResult)).
            plus(secondConditionalFalseNestedTrue.times(rScaled)).
            plus(secondConditionalFalseNestedFalse.times(secondConditionalFalseNestedFalseResult));
    }

    private static DoubleTensor shoulder(DoubleTensor Sw, DoubleTensor Bw, DoubleTensor x) {
        final DoubleTensor A = getCubeCoefficient(Sw, Bw);
        final DoubleTensor B = getSquareCoefficient(Sw, Bw);
        return A.times(x.pow(3)).plus(B.times(x.pow(2)));
    }

    private static DoubleTensor getCubeCoefficient(DoubleTensor Sw, DoubleTensor Bw) {
        return (Sw.pow(3).times(Sw.plus(Bw))).reciprocal().times(-2);
    }

    private static DoubleTensor getSquareCoefficient(DoubleTensor Sw, DoubleTensor Bw) {
        return (Sw.pow(2).times(Sw.plus(Bw))).reciprocal().times(3);
    }

    private static DoubleTensor bodyHeight(DoubleTensor shoulderWidth, DoubleTensor bodyWidth) {
        return shoulderWidth.plus(bodyWidth).reciprocal();
    }

}
