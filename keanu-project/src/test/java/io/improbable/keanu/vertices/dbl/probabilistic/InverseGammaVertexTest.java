package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;

public class InverseGammaVertexTest {

    private static final double DELTA = 0.001;

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();
    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(3);
    }

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 120000;
        double epsilon = 0.01;
        double alpha = 3.0;
        double beta = .5;

        InverseGammaVertex inverted = new InverseGammaVertex(alpha, beta);

        double mean = beta / (alpha - 1.0);
        double standardDeviation = Math.sqrt(Math.pow(beta, 2) / (Math.pow(alpha - 1, 2) * (alpha - 2)));

        ProbabilisticDoubleContract.samplingProducesRealisticMeanAndStandardDeviation(
            N,
            inverted,
            mean,
            standardDeviation,
            epsilon,
            random
        );
    }

    @Test
    public void samplingMatchesLogProb() {
        InverseGammaVertex gamma = new InverseGammaVertex(2.0, 3.0);

        ProbabilisticDoubleContract.sampleMethodMatchesLogProbMethod(
            gamma,
            100000,
            2.0,
            10.0,
            0.1,
            0.01,
            random
        );
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPda() {
        UniformVertex uniformA = new UniformVertex(1.0, 4.0);

        InverseGammaVertex inverted = new InverseGammaVertex(uniformA, 1.0);

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
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdb() {
        UniformVertex uniformB = new UniformVertex(1.0, 3.0);

        InverseGammaVertex inverted = new InverseGammaVertex(2.0, uniformB);

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
    public void isTreatedAsConstantWhenObserved() {
        InverseGammaVertex vertexUnderTest = new InverseGammaVertex(
            new UniformVertex(0.0, 1.0),
            3.0
        );
        ProbabilisticDoubleContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void inferHyperParamsFromSamples() {
        double trueAlpha = 3.0;
        double trueBeta = 0.5;

        List<DoubleVertex> alphaBeta = new ArrayList<>();
        alphaBeta.add(new ConstantDoubleVertex(trueAlpha));
        alphaBeta.add(new ConstantDoubleVertex(trueBeta));

        List<DoubleVertex> latentAlphaBeta = new ArrayList<>();
        latentAlphaBeta.add(new SmoothUniformVertex(0.01, 10.0));
        latentAlphaBeta.add(new SmoothUniformVertex(0.01, 10.0));

        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new InverseGammaVertex(hyperParams.get(0), hyperParams.get(1)),
            alphaBeta,
            latentAlphaBeta,
            10000,
            random
        );
    }
}
