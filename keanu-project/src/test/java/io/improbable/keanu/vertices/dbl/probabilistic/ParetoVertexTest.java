package io.improbable.keanu.vertices.dbl.probabilistic;

import static org.junit.Assert.assertEquals;

import java.util.Map;

import org.apache.commons.math3.distribution.ParetoDistribution;
import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.distributions.gradient.Pareto;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class ParetoVertexTest {
    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {
        ParetoDistribution baseline = new ParetoDistribution(1.0, 1.5);
        ParetoVertex vertex = new ParetoVertex(1.0, 1.5);
        double expected = baseline.logDensity(1.25);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(vertex, 1.25, expected);
    }

    @Test
    public void matchesKnownLogDensityofVector() {
        ParetoDistribution baseline = new ParetoDistribution(1.0, 1.5);
        ParetoVertex vertex = new ParetoVertex(1.0, 1.5);
        double expected = baseline.logDensity(1.25) + baseline.logDensity(6.5);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(vertex, new double[]{1.25, 6.5}, expected);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {
        Pareto.Diff paretoLogDiff = Pareto.dlnPdf(1.0, 1.5, 2.5);

        UniformVertex locationTensor = new UniformVertex(0.0, 1.0);
        locationTensor.setValue(1.0);

        UniformVertex scaleTensor = new UniformVertex(0.0, 2);
        scaleTensor.setValue(1.5);

        ParetoVertex vertex = new ParetoVertex(locationTensor, scaleTensor);
        Map<Long, DoubleTensor> actualDerivatives = vertex.dLogPdf(2.5);
        PartialDerivatives actual = new PartialDerivatives(actualDerivatives);

        assertEquals(paretoLogDiff.dPdXm, actual.withRespectTo(locationTensor.getId()).scalar(), 1e-5);
        assertEquals(paretoLogDiff.dPdAlpha, actual.withRespectTo(scaleTensor.getId()).scalar(), 1e-5);
        assertEquals(paretoLogDiff.dPdX, actual.withRespectTo(vertex.getId()).scalar(), 1e-5);
    }
}
