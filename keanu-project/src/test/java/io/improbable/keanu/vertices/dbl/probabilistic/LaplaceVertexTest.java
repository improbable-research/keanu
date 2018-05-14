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

public class LaplaceVertexTest {

    private static final double DELTA = 0.0001;

    private Random random;

    @Before
    public void setup() {
        random = new Random(1);
    }

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 100000;
        double epsilon = 0.01;
        LaplaceVertex laplaceVertex = new LaplaceVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(1.0), random);

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
        LaplaceVertex laplaceVertex = new LaplaceVertex(
                new ConstantDoubleVertex(0.0),
                new ConstantDoubleVertex(1.0),
                random
        );

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
                DELTA
        );
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdbeta() {
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
                DELTA
        );
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        LaplaceVertex vertexUnderTest = new LaplaceVertex(
                new UniformVertex(0.0, 1.0),
                new ConstantDoubleVertex(3.0),
                random
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
                hyperParams -> new LaplaceVertex(hyperParams.get(0), hyperParams.get(1), random),
                muBeta,
                latentMuBeta,
                1000,
                random
        );
    }
}
