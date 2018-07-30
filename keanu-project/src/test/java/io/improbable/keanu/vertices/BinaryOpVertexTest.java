package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.binary.BinaryOpLambda;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class BinaryOpVertexTest {
    private final Logger log = LoggerFactory.getLogger(BinaryOpVertexTest.class);

    private KeanuRandom random;

    @Before
    public void setup() {
        this.random = new KeanuRandom(1);
    }

    @Test
    public void canSampleFromTwoParents() {
        Flip flip = new Flip(0.5);

        GaussianVertex gaussianVertex = new GaussianVertex(0.0, 1.0);
        BinaryOpLambda<BooleanTensor, DoubleTensor, DoubleTensor> custom = new BinaryOpLambda<>(
            flip, gaussianVertex,
            (BooleanTensor f, DoubleTensor g) ->
                f.setDoubleIf(g, DoubleTensor.scalar(0.0))
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
