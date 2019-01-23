package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static io.improbable.keanu.distributions.hyperparam.Diffs.X;

/**
 * The Smooth Uniform distribution is the usual Uniform distribution with the edges
 * at max and min smoothed by attaching a sigmoid as shoulders.
 * <p>
 * Math:
 * <p>
 * let sigmoid shoulder function be f(x), Sw be shoulder width, Bw be base (max-min) width,
 * and h be the bodyHeight of the base.
 * <p>
 * f(x) = Ax^3 + Bx^2
 * f'(x) = 3Ax^2 + 2Bx
 * integral f = Ax^4/4 + Bx^3/3
 * <p>
 * f(Sw) = h
 * integral of f from 0 to Sw = 1 (area under the curve must be 1)
 * f'(Sw) = 0
 * <p>
 * yields:
 * <p>
 * |  0    3Sw^2    2Sw   |   | h |   | 0 |
 * | -1    Sw^3     Sw^2  | * | A | = | 0 |
 * | Bw    Sw/4     2Sw/3 |   | B |   | 1 |
 * <p>
 * therefore:
 * <p>
 * h = 1 / (Sw + Bw)
 * A = -2 / (Sw^3 * (Sw + Bw))
 * B = 3 / (Sw^3 * Sw^2*Bw)
 */
public class SmoothUniform implements ContinuousDistribution {

    private final DoubleTensor xMin;
    private final DoubleTensor xMax;
    private final double edgeSharpness;

    /**
     * Will return samples between xMin and xMax as well as samples from the left and right shoulder.
     * The width of the shoulder is determined by the edgeSharpness as a percentage of the body width,
     * which is (xMax - xMin).
     *
     * @param xMin          min value from body
     * @param xMax          max value from body
     * @param edgeSharpness sharpness as a percentage of the body width
     * @return       a new ContinuousDistribution object
     */
    public static ContinuousDistribution withParameters(DoubleTensor xMin, DoubleTensor xMax, double edgeSharpness) {
        return new SmoothUniform(xMin, xMax, edgeSharpness);
    }
    private SmoothUniform(DoubleTensor xMin, DoubleTensor xMax, double edgeSharpness) {
        this.xMin = xMin;
        this.xMax = xMax;
        this.edgeSharpness = edgeSharpness;
    }

    @Override
    public DoubleTensor sample(long[] shape, KeanuRandom random) {

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

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor bodyWidth = xMax.minus(xMin);
        final DoubleTensor shoulderWidth = bodyWidth.times(edgeSharpness);
        final DoubleTensor rightCutoff = xMax.plus(shoulderWidth);
        final DoubleTensor leftCutoff = xMin.minus(shoulderWidth);

        DoubleTensor firstConditional = x.getGreaterThanOrEqualToMask(xMin);
        firstConditional = firstConditional.timesInPlace(x.getLessThanOrEqualToMask(xMax));
        final DoubleTensor firstConditionalResult = bodyHeight(shoulderWidth, bodyWidth);

        DoubleTensor secondConditional = x.getLessThanMask(xMin);
        secondConditional = secondConditional.timesInPlace(x.getGreaterThanMask(leftCutoff));
        final DoubleTensor secondConditionalResult = shoulder(shoulderWidth, bodyWidth, x.minus(leftCutoff));

        DoubleTensor thirdConditional = x.getGreaterThanMask(xMax);
        thirdConditional = thirdConditional.timesInPlace(x.getLessThanMask(rightCutoff));
        final DoubleTensor thirdConditionalResult = shoulder(shoulderWidth, bodyWidth, shoulderWidth.minus(x).plusInPlace(xMax));

        return firstConditional.timesInPlace(firstConditionalResult)
            .plusInPlace(secondConditional.timesInPlace(secondConditionalResult))
            .plusInPlace(thirdConditional.timesInPlace(thirdConditionalResult))
            .logInPlace();
    }

    public static DoubleVertex logProbOutput(LogProbGraph.DoublePlaceholderVertex x, LogProbGraph.DoublePlaceholderVertex xMin, LogProbGraph.DoublePlaceholderVertex xMax, double edgeSharpness) {
        final DoubleVertex bodyWidth = xMax.minus(xMin);
        final DoubleVertex shoulderWidth = bodyWidth.times(edgeSharpness);
        final DoubleVertex rightCutoff = xMax.plus(shoulderWidth);
        final DoubleVertex leftCutoff = xMin.minus(shoulderWidth);

        final DoubleVertex firstConditional = x
            .toGreaterThanOrEqualToMask(xMin)
            .times(x.toLessThanOrEqualToMask(xMax));
        final DoubleVertex firstConditionResult = bodyHeightVertex(shoulderWidth, bodyWidth);

        final DoubleVertex secondConditional = x
            .toLessThanMask(xMin)
            .times(x.toGreaterThanMask(leftCutoff));
        final DoubleVertex secondConditionalResult = shoulderVertex(shoulderWidth, bodyWidth, x.minus(leftCutoff));

        final DoubleVertex thirdConditional = x.toGreaterThanMask(xMax)
            .times(x.toLessThanMask(rightCutoff));
        final DoubleVertex thirdConditionalResult = shoulderVertex(shoulderWidth, bodyWidth, shoulderWidth.minus(x).plus(xMax));

        return firstConditional.times(firstConditionResult)
            .plus(secondConditional.times(secondConditionalResult))
            .plus(thirdConditional.times(thirdConditionalResult))
            .log();
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor bodyWidth = xMax.minus(xMin);
        final DoubleTensor shoulderWidth = bodyWidth.times(edgeSharpness);
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

        return new Diffs()
            .put(X, firstConditional.timesInPlace(firstConditionalResult)
                .plusInPlace(secondConditional.timesInPlace(secondConditionalResult)));
    }

    private static DoubleTensor shoulder(DoubleTensor Sw, DoubleTensor Bw, DoubleTensor x) {
        final DoubleTensor A = getCubeCoefficient(Sw, Bw);
        final DoubleTensor B = getSquareCoefficient(Sw, Bw);
        return x.pow(3).timesInPlace(A).plusInPlace(x.pow(2).timesInPlace(B));
    }

    private static DoubleVertex shoulderVertex(DoubleVertex Sw, DoubleVertex Bw, DoubleVertex x) {
        final DoubleVertex A = getCubeCoefficientVertex(Sw, Bw);
        final DoubleVertex B = getSquareCoefficientVertex(Sw, Bw);
        return x.pow(3).times(A).plus(x.pow(2).times(B));
    }

    private static DoubleTensor dShoulder(DoubleTensor Sw, DoubleTensor Bw, DoubleTensor x) {
        final DoubleTensor A = getCubeCoefficient(Sw, Bw);
        final DoubleTensor B = getSquareCoefficient(Sw, Bw);
        return A.timesInPlace(3).timesInPlace(x.pow(2)).plusInPlace(B.timesInPlace(x).timesInPlace(2));
    }

    private static DoubleTensor getCubeCoefficient(DoubleTensor Sw, DoubleTensor Bw) {
        return (Sw.pow(3).timesInPlace(Sw.plus(Bw))).reciprocalInPlace().timesInPlace(-2);
    }

    private static DoubleVertex getCubeCoefficientVertex(DoubleVertex Sw, DoubleVertex Bw) {
        return Sw.pow(3).times(Sw.plus(Bw)).reverseDiv(-2.);
    }

    private static DoubleTensor getSquareCoefficient(DoubleTensor Sw, DoubleTensor Bw) {
        return (Sw.pow(2).timesInPlace(Sw.plus(Bw))).reciprocalInPlace().timesInPlace(3);
    }

    private static DoubleVertex getSquareCoefficientVertex(DoubleVertex Sw, DoubleVertex Bw) {
        return Sw.pow(2).times(Sw.plus(Bw)).reverseDiv(3.);
    }

    private static DoubleTensor bodyHeight(DoubleTensor shoulderWidth, DoubleTensor bodyWidth) {
        return shoulderWidth.plus(bodyWidth).reciprocalInPlace();
    }

    private static DoubleVertex bodyHeightVertex(DoubleVertex shoulderWidth, DoubleVertex bodyWidth) {
        return shoulderWidth.plus(bodyWidth).reverseDiv(1.);
    }

}
