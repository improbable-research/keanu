package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorPartialDerivatives;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertEquals;

public class TensorGaussianVertexTest {

    private KeanuRandom random;

    @Before
    public void setup() {
        System.setProperty("dtype", "double");
        random = new KeanuRandom(1);
    }


    @Test
    public void matchesKnownLogDensityOfScalar() {

        GaussianVertex gaussianVertex = new GaussianVertex(0, 1, new Random(1));

        TensorGaussianVertex ndGaussianVertex = new TensorGaussianVertex(new ConstantTensorVertex(0), new ConstantTensorVertex(1), new KeanuRandom(1));

        double expectedDensity = gaussianVertex.logPdf(0.5);
        double actualDensity = ndGaussianVertex.logPdf(Nd4jDoubleTensor.scalar(0.5));

        assertEquals(expectedDensity, actualDensity, 1e-5);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {

        Random random = new Random(1);
        ConstantDoubleVertex mu = new ConstantDoubleVertex(0.0);
        ConstantDoubleVertex sigma = new ConstantDoubleVertex(1.0);
        GaussianVertex gaussianVertexA = new GaussianVertex(mu, sigma, random);
        GaussianVertex gaussianVertexB = new GaussianVertex(mu, sigma, random);

        double expectedDensity = gaussianVertexA.logPdf(0.25) + gaussianVertexB.logPdf(-0.75);

        TensorGaussianVertex ndGaussianVertex = new TensorGaussianVertex(new ConstantTensorVertex(0), new ConstantTensorVertex(1), new KeanuRandom(1));

        double actualDensity = ndGaussianVertex.logPdf(DoubleTensor.create(new double[]{0.25, -0.75}, new int[]{2, 1}));

        assertEquals(expectedDensity, actualDensity, 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        Random random = new Random(1);
        UniformVertex mu = new UniformVertex(0.0, 1.0, random);
        mu.setValue(0.0);

        UniformVertex sigma = new UniformVertex(0.0, 1.0, random);
        sigma.setValue(1.0);

        GaussianVertex gaussianVertex = new GaussianVertex(mu, sigma, random);
        Map<String, DoubleTensor> expectedDerivatives = gaussianVertex.dLogPdf(0.5);

        KeanuRandom keanuRandom = new KeanuRandom(1);


        TensorUniformVertex muTensor = new TensorUniformVertex(new ConstantTensorVertex(0.0), new ConstantTensorVertex(1.0), keanuRandom);
        muTensor.setValue(Nd4jDoubleTensor.scalar(0.0));

        TensorUniformVertex sigmaTensor = new TensorUniformVertex(new ConstantTensorVertex(0.0), new ConstantTensorVertex(1.0), keanuRandom);
        sigmaTensor.setValue(Nd4jDoubleTensor.scalar(1.0));

        TensorGaussianVertex ndGaussianVertex = new TensorGaussianVertex(muTensor, sigmaTensor, new KeanuRandom(1));
        Map<String, DoubleTensor> actualDerivatives = ndGaussianVertex.dLogPdf(Nd4jDoubleTensor.scalar(0.5));

        assertEquals(expectedDerivatives.get(mu.getId()).scalar(), actualDerivatives.get(muTensor.getId()).scalar(), 1e-5);
        assertEquals(expectedDerivatives.get(sigma.getId()).scalar(), actualDerivatives.get(sigmaTensor.getId()).scalar(), 1e-5);
        assertEquals(expectedDerivatives.get(gaussianVertex.getId()).scalar(), actualDerivatives.get(ndGaussianVertex.getId()).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        Random random = new Random(1);
        UniformVertex mu = new UniformVertex(0.0, 1.0, random);
        mu.setValue(0.0);

        UniformVertex sigma = new UniformVertex(0.0, 1.0, random);
        sigma.setValue(1.0);

        GaussianVertex gaussianVertexA = new GaussianVertex(mu, sigma, random);
        GaussianVertex gaussianVertexB = new GaussianVertex(mu, sigma, random);
        TensorPartialDerivatives expectedDerivativesA = new TensorPartialDerivatives(gaussianVertexA.dLogPdf(0.25));
        TensorPartialDerivatives expectedDerivativesB = new TensorPartialDerivatives(gaussianVertexB.dLogPdf(-0.75));

        TensorPartialDerivatives expected = expectedDerivativesA.add(expectedDerivativesB);

        KeanuRandom keanuRandom = new KeanuRandom(1);
        TensorUniformVertex muTensor = new TensorUniformVertex(new ConstantTensorVertex(0.0), new ConstantTensorVertex(1.0), keanuRandom);
        muTensor.setValue(Nd4jDoubleTensor.scalar(0.0));

        TensorUniformVertex sigmaTensor = new TensorUniformVertex(new ConstantTensorVertex(0.0), new ConstantTensorVertex(1.0), keanuRandom);
        sigmaTensor.setValue(Nd4jDoubleTensor.scalar(1.0));

        TensorGaussianVertex ndGaussianVertex = new TensorGaussianVertex(muTensor, sigmaTensor, new KeanuRandom(1));
        Map<String, DoubleTensor> actualDerivatives = ndGaussianVertex.dLogPdf(
                DoubleTensor.create(new double[]{0.25, -0.75}, new int[]{2, 1})
        );

        TensorPartialDerivatives actual = new TensorPartialDerivatives(actualDerivatives);

        assertEquals(expected.withRespectTo(mu.getId()).scalar(), actualDerivatives.get(muTensor.getId()).scalar(), 1e-5);
        assertEquals(expected.withRespectTo(sigma.getId()).scalar(), actualDerivatives.get(sigmaTensor.getId()).scalar(), 1e-5);
        assertEquals(expected.withRespectTo(gaussianVertexA.getId()).scalar(), actualDerivatives.get(ndGaussianVertex.getId()).getValue(0), 1e-5);
        assertEquals(expected.withRespectTo(gaussianVertexB.getId()).scalar(), actualDerivatives.get(ndGaussianVertex.getId()).getValue(1), 1e-5);
    }

//    @Test
    public void inferHyperParamsFromSamples() {

        double trueMu = 4.5;
        double trueSigma = 2.0;

        List<DoubleTensorVertex> muSigma = new ArrayList<>();
        muSigma.add(new ConstantTensorVertex(Nd4jDoubleTensor.scalar(trueMu)));
        muSigma.add(new ConstantTensorVertex(Nd4jDoubleTensor.scalar(trueSigma)));

        List<DoubleTensorVertex> latentMuSigma = new ArrayList<>();
        latentMuSigma.add(new TensorUniformVertex(0.01, 10.0, random));
        latentMuSigma.add(new TensorUniformVertex(0.01, 10.0, random));

        int numSamples = 10000;
        TensorVertexVariationalMAP.inferHyperParamsFromSamples(
                hyperParams -> new TensorGaussianVertex(new int[]{numSamples, 1}, hyperParams.get(0), hyperParams.get(1), random),
                muSigma,
                latentMuSigma
        );
    }
}
