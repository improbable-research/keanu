package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.gradient.Cauchy;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import org.apache.commons.math3.distribution.CauchyDistribution;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;

public class HalfCauchyVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {

        CauchyDistribution distribution = new CauchyDistribution(0.0, 1.0);
        HalfCauchyVertex tensorHalfCauchyVertex = new HalfCauchyVertex(1);
        double expectedDensity = distribution.logDensity(0.5) + Math.log(2.0);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorHalfCauchyVertex, 0.5, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {

        CauchyDistribution distribution = new CauchyDistribution(0.0, 1.0);
        double expectedLogDensity = distribution.logDensity(0.25) + distribution.logDensity(0.75)  + 2.0 * Math.log(2.0);
        HalfCauchyVertex tensorHalfCauchyVertex = new HalfCauchyVertex(1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorHalfCauchyVertex, new double[]{0.25, 0.75}, expectedLogDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        Cauchy.Diff cauchyLogDiff = Cauchy.dlnPdf(0.0, 1.0, 0.5);

        UniformVertex scaleTensor = new UniformVertex(0.0, 1.0);
        scaleTensor.setValue(1.0);

        HalfCauchyVertex tensorHalfCauchyVertex = new HalfCauchyVertex(scaleTensor);
        Map<Long, DoubleTensor> actualDerivatives = tensorHalfCauchyVertex.dLogPdf(0.5);

        PartialDerivatives actual = new PartialDerivatives(actualDerivatives);

        assertEquals(cauchyLogDiff.dPdscale, actual.withRespectTo(scaleTensor.getId()).scalar(), 1e-5);
        assertEquals(cauchyLogDiff.dPdx, actual.withRespectTo(tensorHalfCauchyVertex.getId()).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        double[] vector = new double[]{0.25, 0.75, 0.1, 2, 1.3};

        UniformVertex scaleTensor = new UniformVertex(0.0, 1.0);
        scaleTensor.setValue(1.0);

        Supplier<HalfCauchyVertex> vertexSupplier = () -> new HalfCauchyVertex(scaleTensor);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, vertexSupplier);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        HalfCauchyVertex vertexUnderTest = new HalfCauchyVertex(3.0);
        vertexUnderTest.setAndCascade(Nd4jDoubleTensor.scalar(1.0));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdscale() {
        UniformVertex uniformA = new UniformVertex(1.5, 3.0);
        HalfCauchyVertex cauchy = new HalfCauchyVertex(uniformA);

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(0.0);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(0.5);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(1.0),
            Nd4jDoubleTensor.scalar(3.0),
            0.1,
            uniformA,
            cauchy,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void cauchySampleMethodMatchesLogProbMethod() {

        int sampleCount = 1000000;
        HalfCauchyVertex vertex = new HalfCauchyVertex(
            new int[]{sampleCount, 1},
            ConstantVertex.of(2.0)
        );

        double from = 0;
        double to = 4;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2, random);
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueScale = 2.0;

        List<DoubleVertex> scale = new ArrayList<>();
        scale.add(ConstantVertex.of(trueScale));

        List<DoubleVertex> latentScaleList = new ArrayList<>();
        UniformVertex latentScale = new UniformVertex(0.01, 10.0);
        latentScale.setAndCascade(DoubleTensor.scalar(0.1));
        latentScaleList.add(latentScale);

        int numSamples = 2000;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new HalfCauchyVertex(new int[]{numSamples, 1}, hyperParams.get(0)),
            scale,
            latentScaleList,
            random
        );
    }
}
