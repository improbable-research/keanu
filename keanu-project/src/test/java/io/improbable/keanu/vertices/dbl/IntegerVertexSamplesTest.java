package io.improbable.keanu.vertices.dbl;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertexSamples;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;

public class IntegerVertexSamplesTest {
    List<IntegerTensor> values = ImmutableList.of(
        IntegerTensor.create(new int[]{0, 16, 4}, 1, 3),
        IntegerTensor.create(new int[]{-4, -8, 4}, 1, 3),
        IntegerTensor.create(new int[]{8, -4, 12}, 1, 3),
        IntegerTensor.create(new int[]{4, 4, 8}, 1, 3)
    );
    final IntegerVertexSamples samples = new IntegerVertexSamples(values);

    List<SummaryStatistics> stats = ImmutableList.of(
        new SummaryStatistics(),
        new SummaryStatistics(),
        new SummaryStatistics()
    );

    @Before
    public void calculateSummaryStatistics() {
        for (IntegerTensor tensor : values) {
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
        assertThat(averages.asFlatDoubleArray(), equalTo(new double[] {2.0, 2.0, 7.0}));
    }

    @Test
    public void doesGetScalarMode() {
        int mode = samples.getScalarMode();
        assertThat(mode, equalTo(0));
    }

    @Test
    public void getModeAtIndexReturnsMostFrequentElement() {
        int mode = samples.getModeAtIndex(0, 2);
        assertThat(mode, equalTo(4));
    }

    @Test
    public void getModeAtIndexReturnsFirstElementIfAllUnique() {
        int mode = samples.getModeAtIndex(0, 1);
        assertThat(mode, equalTo(16));
    }

    @Test
    public void canGetSamplesAsTensor() {
        List<IntegerTensor> samplesAsList = samples.asList();
        IntegerTensor samplesAsTensor = samples.asTensor();

        assertThat(samplesAsTensor.getShape(), equalTo(new long[] {4, 1, 3}));

        List<IntegerTensor> samplesAsTensorSliced = new ArrayList<>();
        for (long i = 0; i < samplesAsTensor.getShape()[0]; i++) {
            samplesAsTensorSliced.add(samplesAsTensor.slice(0, i));
        }

        for (int i = 0; i < samplesAsList.size(); i++) {
            IntegerTensor sampleFromList = samplesAsList.get(i);
            IntegerTensor sampleFromTensor = samplesAsTensorSliced.get(i);

            assertThat(sampleFromList.getShape(), equalTo(sampleFromTensor.getShape()));
            assertThat(sampleFromList.asFlatDoubleArray(), equalTo(sampleFromTensor.asFlatDoubleArray()));
        }
    }
}
