package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class TensorSmoothUniform {

    private TensorSmoothUniform() {
    }

    public static DoubleTensor sample(int[] shape, DoubleTensor xMin, DoubleTensor xMax, double edgeSharpness, KeanuRandom random) {

        DoubleTensor r1 = random.nextDouble(shape);
        DoubleTensor r2 = random.nextDouble(shape);

        final DoubleTensor bodyWidth = xMax.minus(xMin);
        final DoubleTensor shoulderWidth = bodyWidth.times(edgeSharpness);
        final DoubleTensor rScaled = r1.timesInPlace(bodyWidth.plus(shoulderWidth)).plusInPlace(xMin.minus(shoulderWidth.div(2)));
        final DoubleTensor bodyHeight = bodyHeight(shoulderWidth, bodyWidth);

        DoubleTensor firstConditional = rScaled.getGreaterThanOrEqualToMask(xMin);
        firstConditional = firstConditional.timesInPlace(rScaled.getLessThanOrEqualToMask(xMax));
        final DoubleTensor inverseFirstConditional = DoubleTensor.ones(firstConditional.getShape()).minusInPlace(firstConditional);

        final DoubleTensor secondConditional = rScaled.getLessThanMask(xMin);
        DoubleTensor spillOnToShoulder = xMin.minus(rScaled);
        DoubleTensor shoulderX = shoulderWidth.minus(spillOnToShoulder);
        DoubleTensor shoulderDensity = shoulder(shoulderWidth, bodyWidth, shoulderX);
        DoubleTensor acceptProbability = shoulderDensity.div(bodyHeight);

        final DoubleTensor secondConditionalNestedTrue = secondConditional.times(r2.getLessThanOrEqualToMask(acceptProbability));
        final DoubleTensor secondConditionalNestedFalse = secondConditional.timesInPlace(r2.getGreaterThanMask(acceptProbability));
        final DoubleTensor secondConditionalNestedFalseResult = xMin.minus(shoulderWidth).plusInPlace(spillOnToShoulder);

        final DoubleTensor secondConditionalFalse = rScaled.getGreaterThanOrEqualToMask(xMin);
        spillOnToShoulder = rScaled.minus(xMax);
        shoulderX = shoulderWidth.minus(spillOnToShoulder);
        shoulderDensity = shoulder(shoulderWidth, bodyWidth, shoulderX);
        acceptProbability = shoulderDensity.divInPlace(bodyHeight);

        final DoubleTensor secondConditionalFalseNestedTrue = secondConditionalFalse.times(r2.getLessThanOrEqualToMask(acceptProbability));
        final DoubleTensor secondConditionalFalseNestedFalse = secondConditionalFalse.timesInPlace(r2.getGreaterThanMask(acceptProbability));
        final DoubleTensor secondConditionalFalseNestedFalseResult = shoulderWidth.plusInPlace(xMax).minusInPlace(spillOnToShoulder);

        return firstConditional.timesInPlace(rScaled)
            .plusInPlace(inverseFirstConditional.times(secondConditionalNestedTrue).timesInPlace(rScaled))
            .plusInPlace(inverseFirstConditional.times(secondConditionalNestedFalse).timesInPlace(secondConditionalNestedFalseResult))
            .plusInPlace(inverseFirstConditional.times(secondConditionalFalseNestedTrue).timesInPlace(rScaled))
            .plusInPlace(inverseFirstConditional.timesInPlace(secondConditionalFalseNestedFalse).timesInPlace(secondConditionalFalseNestedFalseResult));
    }

    public static DoubleTensor pdf(DoubleTensor xMin, DoubleTensor xMax, DoubleTensor shoulderWidth, DoubleTensor x) {

        final DoubleTensor bodyWidth = xMax.minus(xMin);
        final DoubleTensor rightCutoff = xMax.plus(shoulderWidth);
        final DoubleTensor leftCutoff = xMin.minus(shoulderWidth);

        DoubleTensor firstConditional = x.getGreaterThanOrEqualToMask(xMin);
        firstConditional = firstConditional.timesInPlace(x.getLessThanOrEqualToMask(xMax));
        firstConditional.timesInPlace(x.getLessThanOrEqualToMask(xMax));
        final DoubleTensor firstConditionalResult = bodyHeight(shoulderWidth, bodyWidth);

        DoubleTensor secondConditional = x.getLessThanMask(xMin);
        secondConditional = secondConditional.timesInPlace(x.getGreaterThanMask(leftCutoff));
        final DoubleTensor secondConditionalResult = shoulder(shoulderWidth, bodyWidth, x.minus(leftCutoff));

        DoubleTensor thirdConditional = x.getGreaterThanMask(xMax);
        thirdConditional = thirdConditional.timesInPlace(x.getLessThanMask(rightCutoff));
        final DoubleTensor thirdConditionalResult = shoulder(shoulderWidth, bodyWidth, shoulderWidth.minus(x).plusInPlace(xMax));

        return firstConditional.timesInPlace(firstConditionalResult)
            .plusInPlace(secondConditional.timesInPlace(secondConditionalResult))
            .plusInPlace(thirdConditional.timesInPlace(thirdConditionalResult));
    }

    public static DoubleTensor dlnPdf(DoubleTensor xMin, DoubleTensor xMax, DoubleTensor shoulderWidth, DoubleTensor x) {
        final DoubleTensor bodyWidth = xMax.minus(xMin);
        final DoubleTensor leftCutoff = xMin.minus(shoulderWidth);
        final DoubleTensor rightCutoff = xMax.plus(shoulderWidth);

        DoubleTensor firstConditional = x.getLessThanMask(xMin);
        firstConditional = firstConditional.timesInPlace(x.getGreaterThanMask(leftCutoff));
        final DoubleTensor firstConditionalResult = dShoulder(shoulderWidth, bodyWidth, x.minus(leftCutoff));

        DoubleTensor secondConditional = x.getGreaterThanMask(xMax);
        secondConditional = secondConditional.timesInPlace(x.getLessThanMask(rightCutoff));
        final DoubleTensor secondConditionalResult = dShoulder(shoulderWidth,
            bodyWidth,
            shoulderWidth.minus(x).plusInPlace(rightCutoff)
            ).unaryMinusInPlace();

        return firstConditional.timesInPlace(firstConditionalResult)
            .plusInPlace(secondConditional.timesInPlace(secondConditionalResult));
    }

    private static DoubleTensor shoulder(DoubleTensor Sw, DoubleTensor Bw, DoubleTensor x) {
        final DoubleTensor A = getCubeCoefficient(Sw, Bw);
        final DoubleTensor B = getSquareCoefficient(Sw, Bw);
        return x.pow(3).timesInPlace(A).plusInPlace(x.pow(2).timesInPlace(B));
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
