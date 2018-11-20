package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;


public class StatisticsCalculatorTest {

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    private DoubleTensor xData = DoubleTensor.create(1., 2., 3., 4.);
    private DoubleTensor yData = DoubleTensor.create(0., 1., 2., 3.);
    private StatisticsCalculator stats = StatisticsCalculator.forData(xData, yData);

    @Test
    public void ifYouConstructItWithTensorsTheirShapesMustMatch() {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Shapes must match");

        DoubleTensor smallerData = DoubleTensor.create(1., 2., 3.);
        StatisticsCalculator.forData(xData, smallerData);
    }

    @Test
    public void youCanGetTheNumberOfSamples() {
        assertThat(stats.size(), equalTo(xData.getLength()));
    }

    @Test
    public void youCanGetTheMeanOfX() {
        assertThat(stats.xMean(), equalTo(xData.average()));
    }

    @Test
    public void youCanGetTheMeanOfY() {
        assertThat(stats.yMean(), equalTo(yData.average()));
    }

    @Test
    public void youCanGetTheRegressionCoefficients() {
        long n = xData.getLength();
        double sum_x = xData.sum();
        double xBar = xData.average();
        double sum_xsq = xData.times(xData).sum();
        double s_xx = sum_xsq - (sum_x * sum_x / n);

        double sum_y = yData.sum();
        double yBar = yData.average();
        double sum_xy = xData.times(yData).sum();
        double s_xy = sum_xy - (sum_x * sum_y / n);

        double expectedGradient = s_xy / s_xx;
        assertThat(stats.estimatedGradient(), equalTo(expectedGradient));

        double expectedIntercept = yBar - expectedGradient * xBar;
        assertThat(stats.estimateIntercept(), equalTo(expectedIntercept));

    }
}