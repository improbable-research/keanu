package io.improbable.keanu.network;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsInAnyOrder;

public class LambdaSectionTraversalTest {

    GaussianVertex A;
    GaussianVertex B;
    DoubleVertex aPlusB;
    ConstantDoubleVertex cSigma;
    GaussianVertex C;

    @Before
    public void setup() {
        A = new GaussianVertex(0, 1);
        B = new GaussianVertex(0, 1);
        aPlusB = A.plus(B);
        cSigma = ConstantVertex.of(1.0);
        C = new GaussianVertex(aPlusB, cSigma);
    }

    @Test
    public void doesGetDownstreamProbabilisticVertices() {
        LambdaSection lambdaSection = LambdaSection.getDownstreamLambdaSection(A, false);
        assertThat(lambdaSection.getAllVertices(), containsInAnyOrder(A, C));
    }

    @Test
    public void doesGetAllDownstreamVertices() {
        LambdaSection lambdaSection = LambdaSection.getDownstreamLambdaSection(A, true);
        assertThat(lambdaSection.getAllVertices(), containsInAnyOrder(A, aPlusB, C));
    }

    @Test
    public void doesGetUpstreamProbabilisticVertices() {
        LambdaSection lambdaSection = LambdaSection.getUpstreamLambdaSection(C, false);
        assertThat(lambdaSection.getAllVertices(), containsInAnyOrder(A, B, C));
    }

    @Test
    public void doesGetAllUpstreamVertices() {
        LambdaSection lambdaSection = LambdaSection.getUpstreamLambdaSection(C, true);
        assertThat(lambdaSection.getAllVertices(), containsInAnyOrder(A, B, aPlusB, cSigma, C));
    }

    @Test
    public void doesGetUpstreamProbabilisticVerticesOfCollection() {
        LambdaSection lambdaSection = LambdaSection.getUpstreamLambdaSectionForCollection(Arrays.asList(C, cSigma), false);
        assertThat(lambdaSection.getLatentAndObservedVertices(), containsInAnyOrder(A, B, C));
    }

    @Test
    public void doesGetDownstreamProbabilisticVerticesOfCollection() {
        LambdaSection lambdaSection = LambdaSection.getDownstreamLambdaSectionForCollection(Arrays.asList(A, B), false);
        assertThat(lambdaSection.getAllVertices(), containsInAnyOrder(A, C, B));
    }

    @Test
    public void doesGetAllUpstreamVerticesOfCollection() {
        LambdaSection lambdaSection = LambdaSection.getUpstreamLambdaSectionForCollection(Arrays.asList(C, cSigma), true);
        assertThat(lambdaSection.getAllVertices(), containsInAnyOrder(A, B, aPlusB, cSigma, C));
    }

    @Test
    public void doesGetAllDownstreamVerticesOfCollection() {
        LambdaSection lambdaSection = LambdaSection.getDownstreamLambdaSectionForCollection(Arrays.asList(A, B, cSigma), true);
        assertThat(lambdaSection.getAllVertices(), containsInAnyOrder(A, B, aPlusB, cSigma, C));
    }
}
