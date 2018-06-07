package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.vertices.dbl.KeanuRandom;

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
public class SmoothUniform {

    private SmoothUniform() {

    }

    /**
     * Will return samples between xMin and xMax as well as samples from the left and right shoulder.
     * The width of the shoulder is determined by the edgeSharpness as a percentage of the body width,
     * which is (xMax - xMin).
     *
     * @param xMin          min value from body
     * @param xMax          max value from body
     * @param edgeSharpness sharpness as a percentage of the body width
     * @param random        source of randomness
     * @return a uniform random number between xMin and xMax
     */
    public static double sample(double xMin, double xMax, final double edgeSharpness, KeanuRandom random) {

        final double r1 = random.nextDouble();
        final double bodyWidth = xMax - xMin;
        final double shoulderWidth = edgeSharpness * bodyWidth;
        final double rScaled = r1 * (bodyWidth + shoulderWidth) + (xMin - shoulderWidth / 2);

        if (rScaled >= xMin && rScaled <= xMax) {
            return rScaled;
        }

        final double bodyHeight = bodyHeight(shoulderWidth, bodyWidth);
        final double r2 = random.nextDouble();

        if (rScaled < xMin) {

            double spillOnToShoulder = xMin - rScaled;
            double shoulderX = shoulderWidth - spillOnToShoulder;
            double shoulderDensity = shoulder(shoulderWidth, bodyWidth, shoulderX);
            double acceptProbability = shoulderDensity / bodyHeight;

            if (r2 <= acceptProbability) {
                return rScaled;
            } else {
                return xMin - shoulderWidth + spillOnToShoulder;
            }
        } else {

            double spillOnToShoulder = rScaled - xMax;
            double shoulderX = shoulderWidth - spillOnToShoulder;
            double shoulderDensity = shoulder(shoulderWidth, bodyWidth, shoulderX);
            double acceptProbability = shoulderDensity / bodyHeight;

            if (r2 <= acceptProbability) {
                return rScaled;
            } else {
                return xMax + shoulderWidth - spillOnToShoulder;
            }
        }
    }

    public static double pdf(final double xMin, final double xMax, final double shoulderWidth, final double x) {

        final double bodyWidth = xMax - xMin;

        if (x >= xMin && x <= xMax) {
            //x is in flat region and this is the bodyHeight
            return bodyHeight(shoulderWidth, bodyWidth);
        }

        double leftCutoff = xMin - shoulderWidth;
        if (x < xMin && x > leftCutoff) {
            //x is in left shoulder
            final double nx = x - leftCutoff;
            return shoulder(shoulderWidth, bodyWidth, nx);
        }

        double rightCutoff = xMax + shoulderWidth;
        if (x > xMax && x < rightCutoff) {
            //x is in right shoulder
            final double nx = shoulderWidth - (x - xMax);
            return shoulder(shoulderWidth, bodyWidth, nx);
        }

        //x is not in density bounds
        return 0.0;
    }

    private static double getCubeCoefficient(final double Sw, final double Bw) {
        return -2.0 / (Sw * Sw * Sw * (Sw + Bw));
    }

    private static double getSquareCoefficient(final double Sw, final double Bw) {
        return 3.0 / (Sw * Sw * (Sw + Bw));
    }

    private static double bodyHeight(double shoulderWidth, double bodyWidth) {
        return 1.0 / (shoulderWidth + bodyWidth);
    }

    private static double shoulder(final double Sw, final double Bw, final double x) {
        final double A = getCubeCoefficient(Sw, Bw);
        final double B = getSquareCoefficient(Sw, Bw);
        final double xSquared = x * x;
        return A * xSquared * x + B * xSquared;
    }

    private static double dshoulder(final double Sw, final double Bw, final double x) {
        final double A = getCubeCoefficient(Sw, Bw);
        final double B = getSquareCoefficient(Sw, Bw);
        return 3 * A * x * x + 2 * B * x;
    }
}
