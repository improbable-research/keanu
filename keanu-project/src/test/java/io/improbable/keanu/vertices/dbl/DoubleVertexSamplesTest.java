package io.improbable.keanu.vertices.dbl;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Before;
import org.junit.Test;

import java.util.List;

import static io.improbable.keanu.tensor.TensorMatchers.allCloseTo;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;


public class DoubleVertexSamplesTest {
    List<DoubleTensor> values = ImmutableList.of(
        DoubleTensor.create(new double[]{0, 16, 4}, new long[]{1, 3}),
        DoubleTensor.create(new double[]{-4, -8, 4}, new long[]{1, 3}),
        DoubleTensor.create(new double[]{8, -4, 12}, new long[]{1, 3}),
        DoubleTensor.create(new double[]{4, 4, 8}, new long[]{1, 3})
    );
    final DoubleVertexSamples samples = new DoubleVertexSamples(values);

    List<SummaryStatistics> stats = ImmutableList.of(
        new SummaryStatistics(),
        new SummaryStatistics(),
        new SummaryStatistics()
    );

    @Before
    public void calculateSummaryStatistics() {
        for (DoubleTensor tensor : values) {
            for (int i = 0; i < tensor.getLength(); i++) {
                stats.get(i).addValue(tensor.getValue(i));
            }
        }
    }

    @Test
    public void doesCalculateAverage() {

        DoubleTensor averages = samples.getAverages();

        double[] expectedValues = stats.stream().mapToDouble(SummaryStatistics::getMean).toArray();
        assertThat(averages.asFlatDoubleArray(), equalTo(expectedValues));
        assertThat(averages.asFlatDoubleArray(), equalTo(new double[]{2.0, 2.0, 7.0}));


    }

    @Test
    public void doesCalculateVariance() {

        DoubleTensor variances = samples.getVariances();

        DoubleTensor expectedValues = DoubleTensor.create(
            stats.stream().mapToDouble(SummaryStatistics::getVariance).toArray()
        );
        assertThat(variances, allCloseTo(1e-8, expectedValues));
    }

    @Test
    public void doesCalculateAutocorrelation() {
        List<DoubleTensor> expectedAutocorrelations = ImmutableList.of(
            DoubleTensor.create(new double[]{1, -0.15, -0.3, -0.05}, new long[]{1, 4}),
            DoubleTensor.create(new double[]{1., -0.27380952, -0.30952381, 0.08333333}, new long[]{1, 4}),
            DoubleTensor.create(new double[]{1., -0.02272727, -0.40909091, -0.06818182}, new long[]{1, 4})
        );

        for (int i = 0; i < expectedAutocorrelations.size(); i++) {
            assertThat(samples.getAutocorrelation(0, i), allCloseTo(0.01, expectedAutocorrelations.get(i)));
        }
    }
}
