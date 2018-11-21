package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThan;


public class BivariateDataStatisticsCalculatorTest {

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    private static final DoubleTensor X_DATA = DoubleTensor.create(1., 2., 3., 4.);
    private static final DoubleTensor Y_DATA = DoubleTensor.create(0.1, 1.4, 2.2, 3.1);
    SummaryStatistics xStats = createStatistics(X_DATA);
    SummaryStatistics yStats = createStatistics(Y_DATA);
    SummaryStatistics xyStats = createCombinedStatistics(X_DATA, Y_DATA);

    private BivariateDataStatisticsCalculator stats = new BivariateDataStatisticsCalculator(X_DATA, Y_DATA);

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
        new BivariateDataStatisticsCalculator(biggerData, smallerData);
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
        assertThat(stats.estimatedGradient(), equalTo(expectedGradient));

        double expectedIntercept = yStats.getMean() - expectedGradient * xStats.getMean();
        assertThat(stats.estimatedIntercept(), equalTo(expectedIntercept));

    }

    @Test
    public void youCanCalculateTheMeanSquaredError() {
        assertThat(stats.meanSquaredError(), greaterThan(0.));
    }

    @Test
    public void theMeanSquaredErrorIsZeroWhenTheDataIsPerfectlyCorrelated() {
        BivariateDataStatisticsCalculator perfectStats = new BivariateDataStatisticsCalculator(X_DATA, X_DATA.times(2.).plus(42.));
        assertThat(perfectStats.meanSquaredError(), equalTo(0.));
    }

    @Test
    public void youCanGetTheStandardErrorForTheGradient() {
        assertThat(stats.standardErrorForGradient(), greaterThan(0.));
    }

    @Test
    public void youCanGetTheStandardErrorForTheIntercept() {
        assertThat(stats.standardErrorForIntercept(), greaterThan(0.));
    }

    @Test
    public void theStandardErrorIsZeroWhenTheDataIsPerfectlyCorrelated() {
        BivariateDataStatisticsCalculator perfectStats = new BivariateDataStatisticsCalculator(X_DATA, X_DATA.times(2.).plus(42.));
        assertThat(perfectStats.standardErrorForGradient(), equalTo(0.));
        assertThat(perfectStats.standardErrorForIntercept(), equalTo(0.));
    }

}