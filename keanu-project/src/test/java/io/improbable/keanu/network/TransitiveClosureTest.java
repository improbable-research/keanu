package io.improbable.keanu.network;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsInAnyOrder;

public class TransitiveClosureTest {

    DoubleVertex Amu;
    DoubleVertex Asigma;
    DoubleVertex Bmu;
    DoubleVertex Bsigma;
    GaussianVertex A;
    GaussianVertex B;
    AdditionVertex aPlusB;
    ConstantDoubleVertex cSigma;
    GaussianVertex C;

    @Before
    public void setup() {
        Amu = ConstantVertex.of(0.);
        Asigma = ConstantVertex.of(1.);
        Bmu = ConstantVertex.of(0.);
        Bsigma = ConstantVertex.of(1.);
        A = new GaussianVertex(Amu, Asigma);
        B = new GaussianVertex(Bmu, Bsigma);
        aPlusB = A.plus(B);
        cSigma = ConstantVertex.of(1.0);
        C = new GaussianVertex(aPlusB, cSigma);
    }

    @Test
    public void doesGetDownstreamProbabilisticVertices() {
        TransitiveClosure transitiveClosure = TransitiveClosure.getDownstreamVertices(A, false);
        assertThat(transitiveClosure.getAllVertices(), containsInAnyOrder(A, C));
    }

    @Test
    public void doesGetAllDownstreamVertices() {
        TransitiveClosure transitiveClosure = TransitiveClosure.getDownstreamVertices(A, true);
        assertThat(transitiveClosure.getAllVertices(), containsInAnyOrder(A, aPlusB, C));
    }

    @Test
    public void doesGetUpstreamProbabilisticVertices() {
        TransitiveClosure transitiveClosure = TransitiveClosure.getUpstreamVertices(C, false);
        assertThat(transitiveClosure.getAllVertices(), containsInAnyOrder(A, B, C));
    }

    @Test
    public void doesGetAllUpstreamVertices() {
        TransitiveClosure transitiveClosure = TransitiveClosure.getUpstreamVertices(C, true);
        assertThat(transitiveClosure.getAllVertices(), containsInAnyOrder(Amu, Asigma, Bmu, Bsigma, A, B, aPlusB, cSigma, C));
    }

    @Test
    public void doesGetUpstreamProbabilisticVerticesOfCollection() {
        TransitiveClosure transitiveClosure = TransitiveClosure.getUpstreamVerticesForCollection(Arrays.asList(C, cSigma), false);
        assertThat(transitiveClosure.getLatentAndObservedVertices(), containsInAnyOrder(A, B, C));
    }

    @Test
    public void doesGetDownstreamProbabilisticVerticesOfCollection() {
        TransitiveClosure transitiveClosure = TransitiveClosure.getDownstreamVerticesForCollection(Arrays.asList(A, B), false);
        assertThat(transitiveClosure.getAllVertices(), containsInAnyOrder(A, C, B));
    }

    @Test
    public void doesGetAllUpstreamVerticesOfCollection() {
        TransitiveClosure transitiveClosure = TransitiveClosure.getUpstreamVerticesForCollection(Arrays.asList(C, cSigma), true);
        assertThat(transitiveClosure.getAllVertices(), containsInAnyOrder(Amu, Asigma, Bmu, Bsigma, A, B, aPlusB, cSigma, C));
    }

    @Test
    public void doesGetAllDownstreamVerticesOfCollection() {
        TransitiveClosure transitiveClosure = TransitiveClosure.getDownstreamVerticesForCollection(Arrays.asList(A, B, cSigma), true);
        assertThat(transitiveClosure.getAllVertices(), containsInAnyOrder(A, B, aPlusB, cSigma, C));
    }
}