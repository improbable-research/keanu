package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

public class StatisticsCalculator {

    private final DoubleTensor xData;
    private final DoubleTensor yData;

    public StatisticsCalculator(DoubleTensor xData, DoubleTensor yData) {
        TensorShapeValidation.checkAllShapesMatch(xData.getShape(), yData.getShape());
        this.xData = xData;
        this.yData = yData;
    }

    public long size() {
        return xData.getLength();
    }

    public double xMean() {
        return xData.average();
    }

    public double yMean() {
        return yData.average();
    }

    public double estimatedGradient() {
        return secondMomentOf(xData, yData) / secondMomentOf(xData);
    }

    public double estimatedIntercept() {
        return yMean() - estimatedGradient() * xMean();
    }

    public double meanSquaredError() {
        DoubleTensor calculatedY = xData.times(estimatedGradient()).plusInPlace(estimatedIntercept());
        DoubleTensor residuals = yData.minus(calculatedY);
        long unbiasedMultiplier = size() - 2;
        return residuals.times(residuals).sum() / unbiasedMultiplier;
    }

    public double standardErrorForGradient() {
        return Math.sqrt(meanSquaredError() / secondMomentOf(xData));
    }

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
