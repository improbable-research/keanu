package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;

public class InverseGammaVertexTest {

    private final Logger log = LoggerFactory.getLogger(InverseGammaVertexTest.class);

    private Random random;

    private static final double DELTA = 0.001;

    @Before
    public void setup() {
        random = new Random(3);
    }

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 100000;
        double epsilon = 0.01;
        double alpha = 3.0;
        double beta = .5;

        InverseGammaVertex inverted = new InverseGammaVertex(
                new ConstantDoubleVertex(alpha),
                new ConstantDoubleVertex(beta),
                random
        );

        double mean = beta / (alpha - 1.0);
        double standardDeviation = Math.sqrt(Math.pow(beta, 2) / (Math.pow(alpha - 1, 2) * (alpha - 2)));

        ProbabilisticDoubleContract.samplingProducesRealisticMeanAndStandardDeviation(N, inverted, mean, standardDeviation, epsilon);
    }

    @Test
    public void samplingMatchesPdf() {
        InverseGammaVertex gamma = new InverseGammaVertex(
                new ConstantDoubleVertex(2.0),
                new ConstantDoubleVertex(3.0),
                random
        );

        ProbabilisticDoubleContract.sampleMethodMatchesDensityMethod(
                gamma,
                100000,
                2.0,
                10.0,
                0.1,
                0.01);
    }

    @Test
    public void logDensityIsSameAsLogOfDensity() {
        InverseGammaVertex inverted = new InverseGammaVertex(
                new ConstantDoubleVertex(3.0),
                new ConstantDoubleVertex(0.5),
                random);
        double atValue = 0.5;
        double logOfDensity = Math.log(inverted.density(atValue));
        double logDensity = inverted.logDensity(atValue);
        assertEquals(logDensity, logOfDensity, 0.01);
    }

    @Test
    public void diffLnDensityIsSameAsLogOfDiffDensity() {
        InverseGammaVertex inverted = new InverseGammaVertex(
                new ConstantDoubleVertex(3.0),
                new ConstantDoubleVertex(0.5),
                random);
        ProbabilisticDoubleContract.diffLnDensityIsSameAsLogOfDiffDensity(inverted, 0.5, 0.001);
    }

    @Test
    public void dDensityMatchesFiniteDifferenceCalculationFordPda() {
        UniformVertex uniformA = new UniformVertex(
                new ConstantDoubleVertex(1.0),
                new ConstantDoubleVertex(4.0),
                random);

        InverseGammaVertex inverted = new InverseGammaVertex(
                uniformA,
                new ConstantDoubleVertex(1.0),
                random);

        double vertexStartValue = 0.5;
        double vertexEndValue = 3.0;
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(1.0,
                2.0,
                0.1,
                uniformA,
                inverted,
                vertexStartValue,
                vertexEndValue,
                vertexIncrement,
                DELTA);
    }

    @Test
    public void dDensityMatchesFiniteDifferenceCalculationFordPdb() {
        UniformVertex uniformB = new UniformVertex(
                new ConstantDoubleVertex(1.0),
                new ConstantDoubleVertex(3.0),
                random);

        InverseGammaVertex inverted = new InverseGammaVertex(
                new ConstantDoubleVertex(2.0),
                uniformB,
                random);

        double vertexStartValue = 0.5;
        double vertexEndValue = 3.0;
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(1.0,
                3.0,
                0.1,
                uniformB,
                inverted,
                vertexStartValue,
                vertexEndValue,
                vertexIncrement,
                DELTA);
    }

    @Test
    public void inferHyperParamsFromSamples() {
        double trueAlpha = 3.0;
        double trueBeta = 0.5;

        List<DoubleVertex> alphaBeta = new ArrayList<>();
        alphaBeta.add(new ConstantDoubleVertex(trueAlpha));
        alphaBeta.add(new ConstantDoubleVertex(trueBeta));

        List<DoubleVertex> latentAlphaBeta = new ArrayList<>();
        latentAlphaBeta.add(new SmoothUniformVertex(0.01, 10.0, random));
        latentAlphaBeta.add(new SmoothUniformVertex(0.01, 10.0, random));

        VertexVariationalMAPTest.inferHyperParamsFromSamples(
                hyperParams -> new InverseGammaVertex(hyperParams.get(0), hyperParams.get(1), random),
                alphaBeta,
                latentAlphaBeta,
                10000
        );
    }
}
