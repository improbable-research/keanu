package io.improbable.keanu.vertices;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.binary.BinaryOpLambda;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

@Slf4j
public class BinaryOpVertexTest {

    private KeanuRandom random;

    @Before
    public void setup() {
        this.random = new KeanuRandom(1);
    }

    @Category(Slow.class)
    @Test
    public void canSampleFromTwoParents() {
        BernoulliVertex bernoulliVertex = new BernoulliVertex(0.5);

        GaussianVertex gaussianVertex = new GaussianVertex(0.0, 1.0);
        BinaryOpLambda<BooleanTensor, DoubleTensor, DoubleTensor> custom = new BinaryOpLambda<>(
            bernoulliVertex, gaussianVertex,
            (BooleanTensor f, DoubleTensor g) ->
                f.doubleWhere(g, DoubleTensor.scalar(0.0))
        );

        int N = 1000000;
        List<Double> samples = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            samples.add(custom.sample(random).scalar());
        }

        SummaryStatistics stats = new SummaryStatistics();
        samples.forEach(stats::addValue);

        double mean = stats.getMean();
        double sd = stats.getStandardDeviation();
        log.info("Mean: " + mean);
        log.info("SD: " + sd);
        assertEquals(0.0, mean, 0.01);
        assertEquals(0.707, sd, 0.01);
    }
}
