package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

/**
 * Calculates common statistics of a 2-dimensional data set
 * Treats xData and yData as a series of points in the data set - their shapes must match.
 */
public class StatisticsCalculator {

    private final DoubleTensor xData;
    private final DoubleTensor yData;

    /**
     * @param xData the data set for the predictor variable X
     * @param yData the data set for the dependent variable Y
     */
    public StatisticsCalculator(DoubleTensor xData, DoubleTensor yData) {
        TensorShapeValidation.checkAllShapesMatch(xData.getShape(), yData.getShape());
        this.xData = xData;
        this.yData = yData;
    }

    /**
     * @return the size, i.e. number of points in the data set
     */
    public long size() {
        return xData.getLength();
    }

    /**
     * @return The mean X value from the data set
     */
    public double xMean() {
        return xData.average();
    }

    /**
     * @return The mean Y value from the data set
     */
    public double yMean() {
        return yData.average();
    }

    /**
     * Calculate the estimate of the gradient
     * The regression coefficients (gradient and intercept) can be treated as random variables and estimated from the data.
     * From https://www2.isye.gatech.edu/~yxie77/isye2028/lecture12.pdf
     * @return the estimate of the gradient
     */
    public double estimatedGradient() {
        return secondMomentOf(xData, yData) / secondMomentOf(xData);
    }

    /**
     * Calculate the estimate of the intercept
     * The regression coefficients (gradient and intercept) can be treated as random variables and estimated from the data.
     * From https://www2.isye.gatech.edu/~yxie77/isye2028/lecture12.pdf
     * @return the estimate of the intercept
     */
    public double estimatedIntercept() {
        return yMean() - estimatedGradient() * xMean();
    }

    /**
     * Calculate the MSE (mean squared error) which is an unbiased estimate of the variance of Y
     * @return The MSE
     */
    public double meanSquaredError() {
        DoubleTensor calculatedY = xData.times(estimatedGradient()).plusInPlace(estimatedIntercept());
        DoubleTensor residuals = yData.minus(calculatedY);
        long unbiasedMultiplier = size() - 2;
        return residuals.times(residuals).sum() / unbiasedMultiplier;
    }

    /**
     * Calculate the standard error, which is the unbiased estimate of the standard deviation of the gradient
     * From https://www2.isye.gatech.edu/~yxie77/isye2028/lecture12.pdf
     * @return The standard error
     */
    public double standardErrorForGradient() {
        return Math.sqrt(meanSquaredError() / secondMomentOf(xData));
    }

    /**
     * Calculate the standard error, which is the unbiased estimate of the standard deviation of the intercept
     * From https://www2.isye.gatech.edu/~yxie77/isye2028/lecture12.pdf
     * @return The standard error
     */
    public double standardErrorForIntercept() {
        double value = xMean() * xMean() / secondMomentOf(xData);
        value += 1. / size();
        value *= meanSquaredError();
        return Math.sqrt(value);
    }

    private double secondMomentOf(final DoubleTensor data) {
        return secondMomentOf(data, data);
    }

    private double secondMomentOf(final DoubleTensor data1, final DoubleTensor data2) {
        double sum1 = data1.sum();
        double sum2 = data2.sum();
        double sumOfSquares = data1.times(data2).sum();

        return sumOfSquares - (sum1 * sum2 / size());
    }
}
