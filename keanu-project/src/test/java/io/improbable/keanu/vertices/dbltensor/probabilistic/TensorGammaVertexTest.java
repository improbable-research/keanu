package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GammaVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorPartialDerivatives;
import org.junit.Test;

import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertEquals;

public class TensorGammaVertexTest {

    @Test
    public void matchesKnownLogDensityOfScalar() {
        GammaVertex gamma = new GammaVertex(0.5, 1, 1.5, new Random(1));
        TensorGammaVertex tensorGamma = new TensorGammaVertex(0.5, 1, 1.5, new KeanuRandom(1));

        double expectedDensity = gamma.logPdf(0.5);

        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorGamma, 0.5, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {
        Random random = new Random();

        ConstantDoubleVertex a = new ConstantDoubleVertex(0.5);
        ConstantDoubleVertex theta = new ConstantDoubleVertex(1.);
        ConstantDoubleVertex k = new ConstantDoubleVertex(7.);
        GammaVertex gammaA = new GammaVertex(a, theta, k, random);
        GammaVertex gammaB = new GammaVertex(a, theta, k, random);

        double expectedLogDensity = gammaA.logPdf(1.) + gammaB.logPdf(3.);
        TensorGammaVertex tensorGamma = new TensorGammaVertex(0.5, 1., 7., new KeanuRandom(1));

        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorGamma, new double[]{1., 3.}, expectedLogDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {
        Random random = new Random(1);
        UniformVertex a = new UniformVertex(0.5, 1.0, random);
        a.setValue(0.75);

        UniformVertex theta = new UniformVertex(0.5, 1.0, random);
        theta.setValue(0.75);

        UniformVertex k = new UniformVertex(1, 5.0, random);
        k.setValue(2.5);

        GammaVertex gamma = new GammaVertex(a, theta, k, random);
        Map<Long, DoubleTensor> expectedDerivatives = gamma.dLogPdf(1.5);


        KeanuRandom keanuRandom = new KeanuRandom(1);

        TensorUniformVertex aTensor = new TensorUniformVertex(new ConstantTensorVertex(0.5), new ConstantTensorVertex(1.0), keanuRandom);
        aTensor.setValue(Nd4jDoubleTensor.scalar(0.75));

        TensorUniformVertex thetaTensor = new TensorUniformVertex(new ConstantTensorVertex(0.5), new ConstantTensorVertex(1.0), keanuRandom);
        thetaTensor.setValue(Nd4jDoubleTensor.scalar(0.75));

        TensorUniformVertex kTensor = new TensorUniformVertex(new ConstantTensorVertex(1.0), new ConstantTensorVertex(5.0), keanuRandom);
        kTensor.setValue(Nd4jDoubleTensor.scalar(2.5));

        TensorGammaVertex tensorGamma = new TensorGammaVertex(aTensor, thetaTensor, kTensor, new KeanuRandom(1));
        Map<Long, DoubleTensor> actualDerivatives = tensorGamma.dLogPdf(Nd4jDoubleTensor.scalar(1.5));

        TensorPartialDerivatives actual = new TensorPartialDerivatives(actualDerivatives);

        assertEquals(expectedDerivatives.get(a.getId()).scalar(), actual.withRespectTo(aTensor.getId()).scalar(), 1e-5);
        assertEquals(expectedDerivatives.get(theta.getId()).scalar(), actual.withRespectTo(thetaTensor.getId()).scalar(), 1e-5);
        assertEquals(expectedDerivatives.get(k.getId()).scalar(), actual.withRespectTo(kTensor.getId()).scalar(), 1e-5);
        assertEquals(expectedDerivatives.get(gamma.getId()).scalar(), actual.withRespectTo(tensorGamma.getId()).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        Random random = new Random(1);
        UniformVertex a = new UniformVertex(1.0, 2.0, random);
        a.setValue(1.5);

        UniformVertex theta = new UniformVertex(1.0, 2.0, random);
        theta.setValue(1.5);

        UniformVertex k = new UniformVertex(1.0, 5.0, random);
        k.setValue(3.0);

        GammaVertex gammaA = new GammaVertex(a, theta, k, random);
        GammaVertex gammaB = new GammaVertex(a, theta, k, random);
        TensorPartialDerivatives expectedDerivativesA = new TensorPartialDerivatives(gammaA.dLogPdf(1.5));
        TensorPartialDerivatives expectedDerivativesB = new TensorPartialDerivatives(gammaB.dLogPdf(2.0));

        TensorPartialDerivatives expected = expectedDerivativesA.add(expectedDerivativesB);

        KeanuRandom keanuRandom = new KeanuRandom(1);
        TensorUniformVertex aTensor = new TensorUniformVertex(new ConstantTensorVertex(0.0), new ConstantTensorVertex(1.0), keanuRandom);
        aTensor.setValue(Nd4jDoubleTensor.scalar(1.5));

        TensorUniformVertex thetaTensor = new TensorUniformVertex(new ConstantTensorVertex(0.0), new ConstantTensorVertex(1.0), keanuRandom);
        thetaTensor.setValue(Nd4jDoubleTensor.scalar(1.5));

        TensorUniformVertex kTensor = new TensorUniformVertex(new ConstantTensorVertex(0.0), new ConstantTensorVertex(1.0), keanuRandom);
        kTensor.setValue(Nd4jDoubleTensor.scalar(3.0));

        TensorGammaVertex tensorGammaVertex = new TensorGammaVertex(aTensor, thetaTensor, kTensor, new KeanuRandom(1));
        Map<Long, DoubleTensor> actualDerivatives = tensorGammaVertex.dLogPdf(
            DoubleTensor.create(new double[]{1.5, 2}, new int[]{2, 1})
        );

        TensorPartialDerivatives actual = new TensorPartialDerivatives(actualDerivatives);

        assertEquals(expected.withRespectTo(a.getId()).scalar(), actual.withRespectTo(aTensor.getId()).scalar(), 1e-5);
        assertEquals(expected.withRespectTo(theta.getId()).scalar(), actual.withRespectTo(thetaTensor.getId()).scalar(), 1e-5);
        assertEquals(expected.withRespectTo(k.getId()).scalar(), actual.withRespectTo(kTensor.getId()).scalar(), 1e-5);
        assertEquals(expected.withRespectTo(gammaA.getId()).scalar(), actual.withRespectTo(tensorGammaVertex.getId()).getValue(0), 1e-5);
        assertEquals(expected.withRespectTo(gammaB.getId()).scalar(), actual.withRespectTo(tensorGammaVertex.getId()).getValue(1), 1e-5);
    }


    @Test
    public void isTreatedAsConstantWhenObserved() {
        TensorUniformVertex a = new TensorUniformVertex()
    }

    @Test
    public void gammaSampledMethodMatchesLogProbMethod() {
        KeanuRandom random = new KeanuRandom(1);

        int sampleCount = 1000000;
        TensorGammaVertex vertex = new TensorGammaVertex(
            new int[]{sampleCount, 1},
            new ConstantTensorVertex(1.5),
            new ConstantTensorVertex(2.0),
            new ConstantTensorVertex(7.5),
            random
        );

        double from = 1.5;
        double to = 2.5;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2);
    }

}
