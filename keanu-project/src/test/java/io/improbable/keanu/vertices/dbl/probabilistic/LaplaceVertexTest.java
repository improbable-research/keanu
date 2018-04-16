package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;

public class LaplaceVertexTest {

    private static final double DELTA = 0.0001;

    private final Logger log = LoggerFactory.getLogger(LaplaceVertexTest.class);

    private Random random;

    @Before
    public void setup() {
        random = new Random(1);
    }

    @Test
    public void laplaceSampleMethodMatchesDensityMethod() {
        int N = 100000;
        double epsilon = 0.01;
        LaplaceVertex l = new LaplaceVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(1.0), random);

        List<Double> samples = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            double sample = l.sample();
            samples.add(sample);
        }

        SummaryStatistics stats = new SummaryStatistics();
        samples.forEach(stats::addValue);

        double mean = stats.getMean();
        double sd = stats.getStandardDeviation();
        log.info("Mean: " + mean);
        log.info("Standard deviation: " + sd);
        assertEquals(0.0, mean, epsilon);
        assertEquals(Math.sqrt(2 * 1.0), sd, epsilon);
    }

    @Test
    public void samplingMatchesPdf() {
        LaplaceVertex laplace = new LaplaceVertex(
                new ConstantDoubleVertex(0.0),
                new ConstantDoubleVertex(1.0),
                random
        );

        ProbabilisticDoubleContract.sampleMethodMatchesDensityMethod(
                laplace,
                100000,
                2.0,
                10.0,
                0.1,
                0.01);
    }

    @Test
    public void logDensityIsSameAsLogOfDensity() {
        LaplaceVertex l = new LaplaceVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(1.0), random);
        double atValue = 0.5;
        double logOfDensity = Math.log(l.density(atValue));
        double logDensity = l.logDensity(atValue);
        assertEquals(logDensity, logOfDensity, 0.01);
    }

    @Test
    public void dDensityMatchesFiniteDifferenceCalculationFordPdmu() {
        UniformVertex uniform = new UniformVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(3.0), random);
        LaplaceVertex laplace = new LaplaceVertex(uniform, new ConstantDoubleVertex(1.0), random);

        double vertexStartValue = 2.0;
        double vertexEndValue = 5.0;
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(0.0,
                2.0,
                0.1,
                uniform,
                laplace,
                vertexStartValue,
                vertexEndValue,
                vertexIncrement,
                DELTA);
    }

    @Test
    public void dDensityMatchesFiniteDifferenceCalculationFordPdbeta() {
        UniformVertex uniform = new UniformVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(3.0), random);
        LaplaceVertex laplace = new LaplaceVertex(new ConstantDoubleVertex(0.0), uniform, random);

        double vertexStartValue = -5.0;
        double vertexEndValue = 5.0;
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(2.0,
                3.0,
                0.1,
                uniform,
                laplace,
                vertexStartValue,
                vertexEndValue,
                vertexIncrement,
                DELTA);
    }

    @Test
    public void diffLnDensityIsSameAsLogOfDiffDensity() {
        LaplaceVertex l = new LaplaceVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(1.0), random);
        ProbabilisticDoubleContract.diffLnDensityIsSameAsLogOfDiffDensity(l, 0.5, 0.001);
    }


    @Test
    public void inferHyperParamsFromSamples() {

        double trueMu = 0.0;
        double trueBeta = 1.0;

        List<DoubleVertex> muBeta = new ArrayList<>();
        muBeta.add(new ConstantDoubleVertex(trueMu));
        muBeta.add(new ConstantDoubleVertex(trueBeta));

        List<DoubleVertex> latentMuBeta = new ArrayList<>();
        latentMuBeta.add(new SmoothUniformVertex(0.01, 10.0));
        latentMuBeta.add(new SmoothUniformVertex(0.01, 10.0));

        VertexVariationalMAPTest.inferHyperParamsFromSamples(
                hyperParams -> new LaplaceVertex(hyperParams.get(0), hyperParams.get(1), random),
                muBeta,
                latentMuBeta,
                1000
        );
    }
}
