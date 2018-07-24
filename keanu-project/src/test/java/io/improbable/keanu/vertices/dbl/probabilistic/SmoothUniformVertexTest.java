package io.improbable.keanu.vertices.dbl.probabilistic;

import static org.junit.Assert.assertEquals;

import java.util.Map;

import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class SmoothUniformVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {

        SmoothUniformVertex tensorSmoothUniformVertex = VertexOfType.smoothUniform(0., 1.);
        SmoothUniformVertex smoothUniformVertex = VertexOfType.smoothUniform(0., 1.);
        double expectedDensity = smoothUniformVertex.logProb(DoubleTensor.scalar(0.5));


        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorSmoothUniformVertex, 0.5, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {

        SmoothUniformVertex smoothUniformVertex = VertexOfType.smoothUniform(0., 1.);
        double expectedLogDensity = smoothUniformVertex.logProb(DoubleTensor.scalar(0.25)) + smoothUniformVertex.logProb(DoubleTensor.scalar(0.75));
        SmoothUniformVertex tensorSmoothUniformVertex = VertexOfType.smoothUniform(0., 1.);

        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorSmoothUniformVertex,
            new double[]{0.25, 0.75},
            expectedLogDensity
        );
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        SmoothUniformVertex smoothUniformVertex = VertexOfType.smoothUniform(0., 1., 10.);
        SmoothUniformVertex tensorSmoothUniformVertex = VertexOfType.smoothUniform(0., 1., 10.);

        Map<Long, DoubleTensor> derivativeFlatRegion = smoothUniformVertex.dLogProb(DoubleTensor.scalar(0.5));
        Map<Long, DoubleTensor> tensorDerivativeFlatRegion = tensorSmoothUniformVertex.dLogProb(DoubleTensor.scalar(0.5));

        assertEquals(derivativeFlatRegion.get(smoothUniformVertex.getId()).scalar(),
            tensorDerivativeFlatRegion.get(tensorSmoothUniformVertex.getId()).scalar(),
            DELTA
        );

        Map<Long, DoubleTensor> derivativeLeftRegion = smoothUniformVertex.dLogProb(DoubleTensor.scalar(-0.5));
        Map<Long, DoubleTensor> tensorDerivativeLeftRegion = tensorSmoothUniformVertex.dLogProb(DoubleTensor.scalar(-0.5));

        assertEquals(derivativeLeftRegion.get(smoothUniformVertex.getId()).scalar(),
            tensorDerivativeLeftRegion.get(tensorSmoothUniformVertex.getId()).scalar(),
            DELTA
        );

        Map<Long, DoubleTensor> derivativeRightRegion = smoothUniformVertex.dLogProb(DoubleTensor.scalar(1.5));
        Map<Long, DoubleTensor> tensorDerivativeRightRegion = tensorSmoothUniformVertex.dLogProb(DoubleTensor.scalar(1.5));

        assertEquals(derivativeRightRegion.get(smoothUniformVertex.getId()).scalar(),
            tensorDerivativeRightRegion.get(tensorSmoothUniformVertex.getId()).scalar(),
            DELTA
        );
    }

    @Test
    public void smoothUniformSampleMethodMatchesLogProbMethod() {

        int sampleCount = 1000000;

        double edgeSharpness = 1.0;
        SmoothUniformVertex vertex = new DistributionVertexBuilder()
            .shaped(sampleCount, 1)
            .withInput(ParameterName.MIN, 0.0)
            .withInput(ParameterName.MAX, 1.0)
            .withInput(ParameterName.SHARPNESS, edgeSharpness)
            .smoothUniform();

        double from = -1;
        double to = 2;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex,
            from,
            to,
            bucketSize,
            1e-2,
            random
        );
    }

}
