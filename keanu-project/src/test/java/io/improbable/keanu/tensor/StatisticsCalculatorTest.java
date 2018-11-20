package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;


public class StatisticsCalculatorTest {

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    private static final DoubleTensor X_DATA = DoubleTensor.create(1., 2., 3., 4.);
    private static final DoubleTensor Y_DATA = DoubleTensor.create(0.1, 1.4, 2.2, 3.1);
    SummaryStatistics xStats = createStatistics(X_DATA);
    SummaryStatistics yStats = createStatistics(Y_DATA);
    SummaryStatistics xyStats = createCombinedStatistics(X_DATA, Y_DATA);

    private StatisticsCalculator stats = StatisticsCalculator.forData(X_DATA, Y_DATA);

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

    @Test
    public void ifYouConstructItWithTensorsTheirShapesMustMatch() {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Shapes must match");

        DoubleTensor biggerData = DoubleTensor.create(1., 2., 3., 4., 5., 6.);
        DoubleTensor smallerData = DoubleTensor.create(1., 2., 3.);
        StatisticsCalculator.forData(biggerData, smallerData);
    }

    @Test
    public void youCanGetTheNumberOfSamples() {
        assertThat(stats.size(), equalTo(xStats.getN()));
    }

    @Test
    public void youCanGetTheMeanOfX() {
        assertThat(stats.xMean(), equalTo(xStats.getMean()));
    }

    @Test
    public void youCanGetTheMeanOfY() {
        assertThat(stats.yMean(), equalTo(yStats.getMean()));
    }

    @Test
    public void youCanGetTheRegressionCoefficients() {

        double s_xy = xyStats.getSum() - (xStats.getSum() * yStats.getSum() / xStats.getN());
        double s_xx = xStats.getSecondMoment();

        double expectedGradient = s_xy / s_xx;
        assertThat(stats.estimateGradient(), equalTo(expectedGradient));

        double expectedIntercept = yStats.getMean() - expectedGradient * xStats.getMean();
        assertThat(stats.estimateIntercept(), equalTo(expectedIntercept));

    }

//    @Test
//    public void youCanGetTheMeanSquaredError() {
//        double estimatedIntercept = stats.estimateIntercept();
//        double estimatedGradient = stats.estimateGradient();
//        DoubleTensor calculatedY = X_DATA.times(estimatedGradient).plusInPlace(estimatedIntercept);
//        DoubleTensor residuals = Y_DATA.minus(calculatedY);
//        StatisticsCalculator residualStats = StatisticsCalculator.forData(residuals, residuals);
//        double meanSquaredError = residuals.times(residuals).sum() / (stats.size() - 2);
//        assertThat(stats.getMeanSquaredError(X_DATA, Y_DATA), equalTo(meanSquaredError));
//
//    }
}