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

    public static StatisticsCalculator forData(DoubleTensor xData, DoubleTensor yData) {
        return new StatisticsCalculator(xData, yData);
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

    public double estimateGradient() {
        double sumX = xData.sum();
        double sumY = yData.sum();
        double sumXX = xData.times(xData).sum();
        double sumXY = xData.times(yData).sum();

        double s_xx = sumXX - (sumX * sumX / size());
        double s_xy = sumXY - (sumX * sumY / size());

        return s_xy / s_xx;
    }

    public double estimateIntercept() {
        return yMean() - estimateGradient() * xMean();
    }

    public double meanSquaredError() {
        DoubleTensor calculatedY = xData.times(estimateGradient()).plusInPlace(estimateIntercept());
        DoubleTensor residuals = yData.minus(calculatedY);
        long unbiasedMultiplier = size() - 2;
        return residuals.times(residuals).sum() / unbiasedMultiplier;
    }
}
