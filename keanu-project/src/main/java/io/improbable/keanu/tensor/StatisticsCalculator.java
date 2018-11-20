package io.improbable.keanu.tensor;

import com.google.common.base.Preconditions;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

public class StatisticsCalculator {

    private final SummaryStatistics xStats;
    private final SummaryStatistics yStats;
    private final SummaryStatistics xyStats;

    private StatisticsCalculator(SummaryStatistics xStats, SummaryStatistics yStats, SummaryStatistics xyStats) {
        Preconditions.checkArgument(xStats.getN() == yStats.getN(),
            "xStats and yStats must have the same size",
            xStats.getN(),
            yStats.getN());
        Preconditions.checkArgument(xStats.getN() == xyStats.getN(),
            "xStats and xyStats must have the same size",
            xStats.getN(),
            xyStats.getN());
        this.xStats = xStats;
        this.yStats = yStats;
        this.xyStats = xyStats;

    }

    public static StatisticsCalculator forData(DoubleTensor xData, DoubleTensor yData) {
        SummaryStatistics xStats = createStatistics(xData);
        SummaryStatistics yStats = createStatistics(yData);
        SummaryStatistics xyStats = createCombinedStatistics(xData, yData);
        return new StatisticsCalculator(xStats, yStats, xyStats);
    }

    private static SummaryStatistics createStatistics(DoubleTensor data) {
        SummaryStatistics stats = new SummaryStatistics();
        double[] values = data.asFlatDoubleArray();
        for (int i = 0; i < values.length; i++) {
            stats.addValue(values[i]);
        }
        return stats;
    }

    private static SummaryStatistics createCombinedStatistics(DoubleTensor xData, DoubleTensor yData) {
        TensorShapeValidation.checkAllShapesMatch(xData.getShape(), yData.getShape());
        SummaryStatistics stats = new SummaryStatistics();
        double[] x = xData.asFlatDoubleArray();
        double[] y = yData.asFlatDoubleArray();
        for (int i = 0; i < x.length; i++) {
            stats.addValue(x[i] * y[i]);
        }
        return stats;
    }

    public double calculateMeanSquaredError() {
        return 0.;
    }

    public long size() {
        return xStats.getN();
    }

    public double xMean() {
        return xStats.getMean();
    }

    public double yMean() {
        return yStats.getMean();
    }

    public double estimatedGradient() {
        double s_xy = xyStats.getSum() - (xStats.getSum() * yStats.getSum() / size());
        double s_xx = xStats.getSecondMoment();
        return s_xy / s_xx;
    }

    public double estimateIntercept() {
        return yMean() - estimatedGradient() * xMean();
    }
}
