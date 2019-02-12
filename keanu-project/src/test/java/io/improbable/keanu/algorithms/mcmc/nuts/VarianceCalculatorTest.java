package io.improbable.keanu.algorithms.mcmc.nuts;

import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Test;

import java.util.Map;

import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.ones;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.zeros;
import static org.junit.Assert.assertEquals;

public class VarianceCalculatorTest {

    @Test
    public void canCalculateVarianceOfSamples() {

        DoubleVertex v = new GaussianVertex(0, 1);
        v.setValue(1.0);

        Map<VariableReference, DoubleTensor> position = ImmutableMap.of(v.getReference(), v.getValue());

        KeanuRandom random = new KeanuRandom(0);
        VarianceCalculator varianceCalculator = new VarianceCalculator(zeros(position), ones(position), 0);

        SummaryStatistics statistics = new SummaryStatistics();
        for (int i = 0; i < 100; i++) {

            double s = random.nextGaussian();
            statistics.addValue(s);

            Map<VariableReference, DoubleTensor> sample = ImmutableMap.of(v.getReference(), DoubleTensor.scalar(s));

            varianceCalculator.addSample(sample);

            double variance = varianceCalculator.currentVariance().get(v.getReference()).scalar();
            double expected = statistics.getPopulationVariance();

            assertEquals(expected, variance, 1e-3);
        }
    }
}
