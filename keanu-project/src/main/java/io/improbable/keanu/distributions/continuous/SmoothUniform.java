package io.improbable.keanu.distributions.continuous;

import static io.improbable.keanu.distributions.dual.Diffs.X;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

/**
 * The Smooth Uniform distribution is the usual Uniform distribution with the edges
 * at max and min smoothed by attaching a sigmoid as shoulders.
 * <h4>Math:</h4>
 * Let sigmoid shoulder function be f(x), Sw be shoulder width, Bw be base (max-min) width,
 * and h be the bodyHeight of the base.
 * <p>
 * <a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;Ax^3&space;&plus;&space;Bx^2&space;\\&space;f'(x)&space;=&space;3Ax^2&space;&plus;&space;2Bx&space;\\&space;\int&space;f(x)&space;=&space;\frac{Ax^{4}}{4}&space;&plus;&space;\frac{Bx^3}{3}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;Ax^3&space;&plus;&space;Bx^2&space;\\&space;f'(x)&space;=&space;3Ax^2&space;&plus;&space;2Bx&space;\\&space;\int&space;f(x)&space;=&space;\frac{Ax^{4}}{4}&space;&plus;&space;\frac{Bx^3}{3}" title="f(x) = Ax^3 + Bx^2 \\ f'(x) = 3Ax^2 + 2Bx \\ \int f(x) = \frac{Ax^{4}}{4} + \frac{Bx^3}{3}" /></a>
 * </p>
 * <p>
 * <a href="https://www.codecogs.com/eqnedit.php?latex=f(Sw)&space;=&space;h&space;\\&space;\int_{0}^{1}&space;f(Sw)&space;=&space;1&space;\\&space;f'(Sw)&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(Sw)&space;=&space;h&space;\\&space;\int_{0}^{1}&space;f(Sw)&space;=&space;1&space;\\&space;f'(Sw)&space;=&space;0" title="f(Sw) = h \\ \int_{0}^{1} f(Sw) = 1 \\ f'(Sw) = 0" /></a>
 * </p>
 * <p>
 * (area under the curve must be 1)
 * </p>
 * <p>
 * This yields:
 * <p>
 * <a href="https://www.codecogs.com/eqnedit.php?latex=\begin{bmatrix}&space;0&space;&&space;3Sw^2&space;&&space;2Sw&space;\\&space;-1&space;&&space;Sw^3&space;&&space;Sw^2&space;\\&space;Bw&space;&&space;\frac{Sw}{4}&space;&&space;\frac{2Sw}{3}&space;\end{bmatrix}&space;\cdot&space;\begin{bmatrix}&space;h&space;\\&space;A&space;\\&space;B&space;\end{bmatrix}&space;=&space;\begin{bmatrix}&space;0&space;\\&space;0&space;\\&space;1&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;0&space;&&space;3Sw^2&space;&&space;2Sw&space;\\&space;-1&space;&&space;Sw^3&space;&&space;Sw^2&space;\\&space;Bw&space;&&space;\frac{Sw}{4}&space;&&space;\frac{2Sw}{3}&space;\end{bmatrix}&space;\cdot&space;\begin{bmatrix}&space;h&space;\\&space;A&space;\\&space;B&space;\end{bmatrix}&space;=&space;\begin{bmatrix}&space;0&space;\\&space;0&space;\\&space;1&space;\end{bmatrix}" title="\begin{bmatrix} 0 & 3Sw^2 & 2Sw \\ -1 & Sw^3 & Sw^2 \\ Bw & \frac{Sw}{4} & \frac{2Sw}{3} \end{bmatrix} \cdot \begin{bmatrix} h \\ A \\ B \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}" /></a>
 * </p>
 * <p>
 * Therefore:
 * </p>
 * <p>
 * <a href="https://www.codecogs.com/eqnedit.php?latex=h&space;=&space;\frac{1}{Sw&space;&plus;&space;Bw}&space;\\&space;A&space;=&space;\frac{-2}{Sw^3&space;(Sw&space;&plus;&space;Bw)}&space;\\&space;B&space;=&space;\frac{3}{Sw^3&space;(Sw^2)(Bw)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h&space;=&space;\frac{1}{Sw&space;&plus;&space;Bw}&space;\\&space;A&space;=&space;\frac{-2}{Sw^3&space;(Sw&space;&plus;&space;Bw)}&space;\\&space;B&space;=&space;\frac{3}{Sw^3&space;(Sw^2)(Bw)}" title="h = \frac{1}{Sw + Bw} \\ A = \frac{-2}{Sw^3 (Sw + Bw)} \\ B = \frac{3}{Sw^3 (Sw^2)(Bw)}" /></a>
 * </p>
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
    public DoubleTensor sample(int[] shape, KeanuRandom random) {

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
