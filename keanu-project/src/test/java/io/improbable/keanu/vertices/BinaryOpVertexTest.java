package io.improbable.keanu.vertices;

import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.binary.BinaryOpLambda;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;

public class BinaryOpVertexTest {
    private final Logger log = LoggerFactory.getLogger(BinaryOpVertexTest.class);

    @Test
    public void basicTest() {
        Random r = new Random(1);
        Flip flip = new Flip(new ConstantDoubleVertex(0.5), r);
        GaussianVertex gaus = new GaussianVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(1.0), r);
        BinaryOpLambda<Boolean, Double, Double> custom = new BinaryOpLambda<>(flip, gaus, (Boolean f, Double g) -> {
            if (f) {
                return g;
            } else {
                return 0.0;
            }
        });

        int N = 1000000;
        List<Double> samples = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            samples.add(custom.sample());
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
