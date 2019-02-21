package io.improbable.keanu.algorithms.sampling;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.ProbabilisticModel;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleBinaryOpVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDouble;
import lombok.extern.slf4j.Slf4j;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@Slf4j
public class ForwardSamplerTest {

    @Rule
    public ExpectedException thrown= ExpectedException.none();

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void throwIfSampleFromHasObservationsDownstreamOfLatents() {
        GaussianVertex A = new GaussianVertex(100.0, 1);
        DoubleVertex B = A.plus(ConstantVertex.of(5.));
        GaussianVertex C = new GaussianVertex(B, 1.);
        C.observe(100.);

        ProbabilisticModel model = new KeanuProbabilisticModel(A.getConnectedGraph());

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("Forward sampler cannot be ran if observed variables have a random variable in their upstream lambda section");
        Keanu.Sampling.Forward.withDefaultConfig(random).getPosteriorSamples(model, Arrays.asList(A, B), 1000);
    }

    @Test
    public void canSampleFromPrior() {
        GaussianVertex A = new GaussianVertex(100.0, 1);
        DoubleVertex B = A.plus(ConstantVertex.of(5.));

        ProbabilisticModel model = new KeanuProbabilisticModel(A.getConnectedGraph());

        final int sampleCount = 1000;
        NetworkSamples samples = Keanu.Sampling.Forward.withDefaultConfig(random).getPosteriorSamples(model, Arrays.asList(A, B), sampleCount);

        double averageA = samples.getDoubleTensorSamples(A).getAverages().scalar();
        double averageB = samples.getDoubleTensorSamples(B).getAverages().scalar();

        assertEquals(sampleCount, samples.size());
        assertEquals(100.0, averageA, 0.1);
        assertEquals(105.0, averageB, 0.1);
    }

    @Test
    public void canCalculateProbabilityOfSamples() {
        GaussianVertex A = new GaussianVertex(100.0, 1);
        DoubleVertex B = A.plus(ConstantVertex.of(5.));

        ProbabilisticModel model = new KeanuProbabilisticModel(A.getConnectedGraph());

        final int sampleCount = 1000;
        NetworkSamples samples = Keanu.Sampling.Forward.builder().random(random).calculateSampleProbability(true).build().getPosteriorSamples(model, Arrays.asList(A, B), sampleCount);

        assertTrue(samples.getLogOfMasterP(0) != 0.);
        assertTrue(samples.getLogOfMasterP(1) != 0.);
        assertTrue(samples.getLogOfMasterP(0) != samples.getLogOfMasterP(1));
    }

    @Test
    public void nonProbabilisticVerticesAreRecomputedDuringForwardSample() {
        GaussianVertex A = mock(GaussianVertex.class);
        PowerVertex B = mock(PowerVertex.class);

        when(A.getChildren()).thenReturn(Collections.singleton(B));
        when(A.getId()).thenReturn(new VertexId());
        when(A.sample()).thenReturn(DoubleTensor.scalar(1.));
        when(A.getConnectedGraph()).thenReturn(new HashSet<>(Arrays.asList(A, B)));

        when(B.getId()).thenReturn(new VertexId());
        when(B.calculate()).thenReturn(DoubleTensor.scalar(0.));

        ProbabilisticModel model = new KeanuProbabilisticModel(Arrays.asList(A, B));

        Keanu.Sampling.Forward.withDefaultConfig(random).getPosteriorSamples(model, Arrays.asList(A, B), 100);

        verify(B, times(100)).setValue(DoubleTensor.scalar(0.));
    }

    @Test
    public void samplesInTopologicalOrder() {
        ConstantDoubleVertex A = ConstantVertex.of(1.0);
        ConstantDoubleVertex B = ConstantVertex.of(2.0);
        ConstantDoubleVertex C = ConstantVertex.of(3.0);

        ArrayList<VertexId> ids = new ArrayList<>();

        IDTrackerVertex trackerOne = new IDTrackerVertex(A, B, ids);
        IDTrackerVertex trackerTwo = new IDTrackerVertex(B, C, ids);

        NonProbabilisticIDTrackerVertex trackerThree = new NonProbabilisticIDTrackerVertex(trackerOne, trackerTwo, ids);
        NonProbabilisticIDTrackerVertex trackerFour = new NonProbabilisticIDTrackerVertex(trackerThree, ConstantVertex.of(5.0), ids);

        IDTrackerVertex trackerFive = new IDTrackerVertex(trackerTwo, trackerFour, ids);

        ProbabilisticModel model = new KeanuProbabilisticModel(A.getConnectedGraph());

        Keanu.Sampling.Forward.withDefaultConfig(random).getPosteriorSamples(model, Arrays.asList(trackerThree, trackerOne, trackerTwo, trackerFour, trackerFive), 1);

        assertEquals(trackerOne.getId(), ids.get(0));
        assertEquals(trackerTwo.getId(), ids.get(1));
        assertEquals(trackerThree.getId(), ids.get(2));
        assertEquals(trackerFour.getId(), ids.get(3));
        assertEquals(trackerFive.getId(), ids.get(4));
    }

    @Test
    public void samplesFromRandomVariablesThatAreInUpstreamOfSampleFrom() {
        ConstantDoubleVertex A = ConstantVertex.of(1.0);
        ConstantDoubleVertex B = ConstantVertex.of(2.0);
        ConstantDoubleVertex C = ConstantVertex.of(3.0);

        ArrayList<VertexId> ids = new ArrayList<>();

        IDTrackerVertex trackerOne = new IDTrackerVertex(A, B, ids);
        IDTrackerVertex trackerTwo = new IDTrackerVertex(B, C, ids);

        IDTrackerVertex trackerThree = new IDTrackerVertex(trackerOne, trackerTwo, ids);

        ProbabilisticModel model = new KeanuProbabilisticModel(A.getConnectedGraph());

        Keanu.Sampling.Forward.withDefaultConfig(random).getPosteriorSamples(model, Collections.singletonList(trackerThree), 1);

        assertEquals(trackerOne.getId(), ids.get(0));
        assertEquals(trackerTwo.getId(), ids.get(1));
        assertEquals(trackerThree.getId(), ids.get(2));
    }

    @Test
    public void doesNotSampleFromNonProbabilisticVerticesThatAreBeforeRandomVariablesInTheGraph() {
        ConstantDoubleVertex A = ConstantVertex.of(1.0);
        ConstantDoubleVertex B = ConstantVertex.of(2.0);
        ConstantDoubleVertex C = ConstantVertex.of(3.0);

        ArrayList<VertexId> ids = new ArrayList<>();

        NonProbabilisticIDTrackerVertex trackerOne = new NonProbabilisticIDTrackerVertex(A, B, ids);
        NonProbabilisticIDTrackerVertex trackerTwo = new NonProbabilisticIDTrackerVertex(B, C, ids);

        IDTrackerVertex trackerThree = new IDTrackerVertex(trackerOne, trackerTwo, ids);
        IDTrackerVertex trackerFour = new IDTrackerVertex(trackerThree, ConstantVertex.of(5.0), ids);

        NonProbabilisticIDTrackerVertex trackerFive = new NonProbabilisticIDTrackerVertex(trackerThree, trackerFour, ids);

        ProbabilisticModel model = new KeanuProbabilisticModel(A.getConnectedGraph());

        Keanu.Sampling.Forward.withDefaultConfig(random).getPosteriorSamples(model, Arrays.asList(trackerThree, trackerFour, trackerFive), 1);

        assertEquals(trackerThree.getId(), ids.get(0));
        assertEquals(trackerFour.getId(), ids.get(1));
        assertEquals(trackerFive.getId(), ids.get(2));
    }

    @Test
    public void doesNotSampleFromLatentsThatAreNotInIntersectionOfTransitiveClosureOfSampleFromAndDownstreamOfLatents() {
        ArrayList<VertexId> ids = new ArrayList<>();

        IDTrackerVertex A = new IDTrackerVertex(ConstantVertex.of(5.0), ConstantVertex.of(5.0), ids);
        IDTrackerVertex B = new IDTrackerVertex(ConstantVertex.of(5.0), ConstantVertex.of(5.0), ids);
        IDTrackerVertex C = new IDTrackerVertex(ConstantVertex.of(5.0), ConstantVertex.of(5.0), ids);

        IDTrackerVertex D = new IDTrackerVertex(A, B, ids);
        IDTrackerVertex E = new IDTrackerVertex(B, C, ids);

        NonProbabilisticIDTrackerVertex F = new NonProbabilisticIDTrackerVertex(D, ConstantVertex.of(5.), ids);

        ProbabilisticModel model = new KeanuProbabilisticModel(A.getConnectedGraph());

        Keanu.Sampling.Forward.withDefaultConfig(random).getPosteriorSamples(model, Arrays.asList(F), 1);

        assertEquals(A.getId(), ids.get(0));
        assertEquals(B.getId(), ids.get(1));
        assertEquals(D.getId(), ids.get(2));
        assertEquals(F.getId(), ids.get(3));

        assertEquals(4, ids.size());
    }

    @Test
    public void canStreamSamples() {

        int sampleCount = 1000;
        int dropCount = 100;
        int downSampleInterval = 1;
        GaussianVertex A = new GaussianVertex(0, 1);
        KeanuProbabilisticModel model = new KeanuProbabilisticModel(A.getConnectedGraph());
        Forward algo = Keanu.Sampling.Forward.withDefaultConfig(random);

        double averageA = algo.generatePosteriorSamples(model, model.getLatentVariables())
            .dropCount(dropCount)
            .downSampleInterval(downSampleInterval)
            .stream()
            .limit(sampleCount)
            .mapToDouble(networkState -> networkState.get(A).scalar())
            .average().getAsDouble();

        assertEquals(0.0, averageA, 0.1);
    }

    public static class IDTrackerVertex extends DoubleVertex implements Differentiable, ProbabilisticDouble {

        private final List<VertexId> ids;

        public IDTrackerVertex(@LoadVertexParam("left")DoubleVertex left, @LoadVertexParam("right")DoubleVertex right, List<VertexId> ids) {
            super(left.getShape());
            this.ids = ids;
            setParents(left, right);
        }

        @Override
        public DoubleTensor sample(KeanuRandom random) {
            ids.add(this.getId());
            return DoubleTensor.ZERO_SCALAR;
        }

        @Override
        public double logProb(DoubleTensor value) {
            return 0;
        }

        @Override
        public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor atValue, Set<? extends Vertex> withRespectTo) {
            return null;
        }
    }

    public static class NonProbabilisticIDTrackerVertex extends DoubleBinaryOpVertex implements Differentiable {

        private final List<VertexId> ids;

        public NonProbabilisticIDTrackerVertex(@LoadVertexParam("left")DoubleVertex left, @LoadVertexParam("right")DoubleVertex right, List<VertexId> ids) {
            super(left, right);
            this.ids = ids;
        }

        @Override
        protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
            ids.add(this.getId());
            return DoubleTensor.ZERO_SCALAR;
        }
    }

}
