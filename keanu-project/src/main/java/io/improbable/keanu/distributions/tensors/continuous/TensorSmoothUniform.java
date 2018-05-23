package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class TensorSmoothUniform {

    private TensorSmoothUniform() {
    }

    public static DoubleTensor sample(int[] shape, DoubleTensor xMin, DoubleTensor xMax, double edgeSharpness, KeanuRandom random) {

        final DoubleTensor r1 = random.nextDouble(shape);
        final DoubleTensor bodyWidth = xMax.minus(xMin);
        final DoubleTensor shoulderWidth = bodyWidth.times(edgeSharpness);
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

    public static DoubleTensor pdf(DoubleTensor xMin, DoubleTensor xMax, DoubleTensor shoulderWidth, DoubleTensor x) {

        final DoubleTensor bodyWidth = xMax.minus(xMin);
        final DoubleTensor rightCutoff = xMax.plus(shoulderWidth);
        final DoubleTensor leftCutoff = xMin.minus(shoulderWidth);

        final DoubleTensor firstConditional = x.getGreaterThanOrEqualToMask(xMin);
        firstConditional.timesInPlace(x.getLessThanOrEqualToMask(xMax));
        final DoubleTensor firstConditionalResult = bodyHeight(shoulderWidth, bodyWidth);

        final DoubleTensor secondConditional = x.getLessThanMask(xMin);
        secondConditional.timesInPlace(x.getGreaterThanMask(leftCutoff));
        final DoubleTensor secondConditionalResult = shoulder(shoulderWidth, bodyWidth, x.minus(leftCutoff));

        final DoubleTensor thirdConditional = x.getGreaterThanMask(xMax);
        thirdConditional.timesInPlace(x.getLessThanMask(rightCutoff));
        final DoubleTensor thirdConditionalResult = shoulder(shoulderWidth, bodyWidth, shoulderWidth.minus(x).plus(xMax));

        return firstConditional.times(firstConditionalResult).
            plus(secondConditional.times(secondConditionalResult)).
            plus(thirdConditional.times(thirdConditionalResult));
    }

    public static DoubleTensor dlnPdf(DoubleTensor xMin, DoubleTensor xMax, DoubleTensor shoulderWidth, DoubleTensor x) {
        final DoubleTensor bodyWidth = xMax.minus(xMin);
        final DoubleTensor leftCutoff = xMin.minus(shoulderWidth);
        final DoubleTensor rightCutoff = xMax.plus(shoulderWidth);

        final DoubleTensor firstConditional = x.getLessThanMask(xMin);
        firstConditional.timesInPlace(x.getGreaterThanMask(leftCutoff));
        final DoubleTensor firstConditionalResult = dShoulder(shoulderWidth, bodyWidth, x.minus(leftCutoff));

        final DoubleTensor secondConditional = x.getGreaterThanMask(xMax);
        secondConditional.times(x.getGreaterThanMask(rightCutoff));
        final DoubleTensor secondConditionalResult = dShoulder(shoulderWidth, bodyWidth, shoulderWidth.minus(x).plus(rightCutoff)).unaryMinus();

        return firstConditional.times(firstConditionalResult).
            plus(secondConditional.times(secondConditionalResult));
    }

    private static DoubleTensor shoulder(DoubleTensor Sw, DoubleTensor Bw, DoubleTensor x) {
        final DoubleTensor A = getCubeCoefficient(Sw, Bw);
        final DoubleTensor B = getSquareCoefficient(Sw, Bw);
        return A.times(x.pow(3)).plus(B.times(x.pow(2)));
    }

    private static DoubleTensor dShoulder(DoubleTensor Sw, DoubleTensor Bw, DoubleTensor x) {
        final DoubleTensor A = getCubeCoefficient(Sw, Bw);
        final DoubleTensor B = getSquareCoefficient(Sw, Bw);
        return A.times(3).times(x.pow(2)).plus(B.times(x).times(2));
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
