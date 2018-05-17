package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;

public class LaplaceVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 100000;
        double epsilon = 0.01;
        LaplaceVertex laplaceVertex = new LaplaceVertex(0.0, 1.0);

        double mean = 0.0;
        double standardDeviation = Math.sqrt(2);

        ProbabilisticDoubleContract.samplingProducesRealisticMeanAndStandardDeviation(
            N,
            laplaceVertex,
            mean,
            standardDeviation,
            epsilon,
            random
        );
    }

    @Test
    public void samplingMatchesLogProb() {
        LaplaceVertex laplaceVertex = new LaplaceVertex(0.0, 1.0);

        ProbabilisticDoubleContract.sampleMethodMatchesLogProbMethod(
            laplaceVertex,
            100000,
            2.0,
            10.0,
            0.1,
            0.01,
            random
        );
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdmu() {
        UniformVertex uniform = new UniformVertex(0.0, 3.0);
        LaplaceVertex laplace = new LaplaceVertex(uniform, 1.0);

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
            DELTA
        );
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdbeta() {
        UniformVertex uniform = new UniformVertex(0.0, 3.0);
        LaplaceVertex laplace = new LaplaceVertex(0.0, uniform);

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
            DELTA
        );
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        LaplaceVertex vertexUnderTest = new LaplaceVertex(
            new UniformVertex(0.0, 1.0),
            3.0
        );
        ProbabilisticDoubleContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
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

        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new LaplaceVertex(hyperParams.get(0), hyperParams.get(1)),
            muBeta,
            latentMuBeta,
            1000,
            random
        );
    }
}
