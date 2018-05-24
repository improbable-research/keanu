package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class TensorSmoothUniform {

    private TensorSmoothUniform() {
    }

    public static DoubleTensor sample(int[] shape, DoubleTensor xMin, DoubleTensor xMax, double edgeSharpness, KeanuRandom random) {

        DoubleTensor r1 = random.nextDouble(shape);
        DoubleTensor r2 = random.nextDouble(shape);

        final DoubleTensor bodyWidth = xMax.minus(xMin);
        final DoubleTensor shoulderWidth = bodyWidth.times(edgeSharpness);
        final DoubleTensor rScaled = r1.times(bodyWidth.plus(shoulderWidth)).plusInPlace(xMin.minus(shoulderWidth.div(2)));
        final DoubleTensor bodyHeight = bodyHeight(shoulderWidth, bodyWidth);

        final DoubleTensor firstConditional = rScaled.getGreaterThanOrEqualToMask(xMin);
        firstConditional.timesInPlace(rScaled.getLessThanOrEqualToMask(xMax));
        final DoubleTensor inverseFirstConditional = DoubleTensor.ones(firstConditional.getShape()).minusInPlace(firstConditional);

        final DoubleTensor secondConditional = rScaled.getLessThanMask(xMin);
        DoubleTensor spillOnToShoulder = xMin.minus(rScaled);
        DoubleTensor shoulderX = shoulderWidth.minus(spillOnToShoulder);
        DoubleTensor shoulderDensity = shoulder(shoulderWidth, bodyWidth, shoulderX);
        DoubleTensor acceptProbability = shoulderDensity.div(bodyHeight);

        final DoubleTensor secondConditionalNestedTrue = secondConditional.times(r2.getLessThanOrEqualToMask(acceptProbability));
        final DoubleTensor secondConditionalNestedFalse = secondConditional.times(r2.getGreaterThanMask(acceptProbability));
        final DoubleTensor secondConditionalNestedFalseResult = xMin.minus(shoulderWidth).plus(spillOnToShoulder);

        final DoubleTensor secondConditionalFalse = rScaled.getGreaterThanOrEqualToMask(xMin);
        spillOnToShoulder = rScaled.minus(xMax);
        shoulderX = shoulderWidth.minus(spillOnToShoulder);
        shoulderDensity = shoulder(shoulderWidth, bodyWidth, shoulderX);
        acceptProbability = shoulderDensity.divInPlace(bodyHeight);

        final DoubleTensor secondConditionalFalseNestedTrue = secondConditionalFalse.times(r2.getLessThanOrEqualToMask(acceptProbability));
        final DoubleTensor secondConditionalFalseNestedFalse = secondConditionalFalse.times(r2.getGreaterThanMask(acceptProbability));
        final DoubleTensor secondConditionalFalseNestedFalseResult = xMax.plus(shoulderWidth).minusInPlace(spillOnToShoulder);

        return firstConditional.times(rScaled).
            plus(inverseFirstConditional.times(secondConditionalNestedTrue).times(rScaled)).
            plus(inverseFirstConditional.times(secondConditionalNestedFalse).times(secondConditionalNestedFalseResult)).
            plus(inverseFirstConditional.times(secondConditionalFalseNestedTrue).times(rScaled)).
            plus(inverseFirstConditional.times(secondConditionalFalseNestedFalse).times(secondConditionalFalseNestedFalseResult));
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
        final DoubleTensor thirdConditionalResult = shoulder(shoulderWidth, bodyWidth, shoulderWidth.minus(x).plusInPlace(xMax));

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
        final DoubleTensor secondConditionalResult = dShoulder(shoulderWidth,
            bodyWidth,
            shoulderWidth.minus(x).plus(rightCutoff)
        ).unaryMinus();

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
        return A.timesInPlace(3).timesInPlace(x.pow(2)).plusInPlace(B.timesInPlace(x).timesInPlace(2));
    }

    private static DoubleTensor getCubeCoefficient(DoubleTensor Sw, DoubleTensor Bw) {
        return (Sw.pow(3).timesInPlace(Sw.plus(Bw))).reciprocalInPlace().timesInPlace(-2);
    }

    private static DoubleTensor getSquareCoefficient(DoubleTensor Sw, DoubleTensor Bw) {
        return (Sw.pow(2).timesInPlace(Sw.plus(Bw))).reciprocalInPlace().timesInPlace(3);
    }

    private static DoubleTensor bodyHeight(DoubleTensor shoulderWidth, DoubleTensor bodyWidth) {
        return shoulderWidth.plus(bodyWidth).reciprocalInPlace();
    }

}
