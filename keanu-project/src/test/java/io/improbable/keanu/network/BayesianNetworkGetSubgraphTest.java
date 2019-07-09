package io.improbable.keanu.network;

import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.vertices.IVertex;
import org.junit.BeforeClass;
import org.junit.Test;
import org.mockito.Mockito;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.mockito.Mockito.when;

public class BayesianNetworkGetSubgraphTest {

    private static IVertex v;
    private static BayesianNetwork network;
    private static IVertex grandChild;
    private static IVertex child;
    private static IVertex parent;
    private static IVertex grandParent;

    @BeforeClass
    public static void setUp() {
        v = Mockito.mock(IVertex.class);

        Set<IVertex> parents = new HashSet<>();
        parent = Mockito.mock(IVertex.class);
        parents.add(parent);
        Set<IVertex> children = new HashSet<>();
        child = Mockito.mock(IVertex.class);
        children.add(child);
        when(v.getParents()).thenReturn(parents);
        when(v.getChildren()).thenReturn(children);

        Set<IVertex> grandParents = new HashSet<>();
        grandParent = Mockito.mock(IVertex.class);
        grandParents.add(grandParent);
        Set<IVertex> grandChildren = new HashSet<>();
        grandChild = Mockito.mock(IVertex.class);
        grandChildren.add(grandChild);
        when(parent.getParents()).thenReturn(grandParents);
        when(child.getChildren()).thenReturn(grandChildren);

        when(v.getConnectedGraph()).thenReturn(ImmutableSet.of(v, parent, child, grandChild, grandParent));

        network = new BayesianNetwork(v.getConnectedGraph());
    }

    @Test
    public void aSubgraphOfDegreeZeroContainsOneVertex() {
        Set<IVertex> degree0SubGraph = network.getSubgraph(v, 0);
        assertThat(degree0SubGraph, contains(v));
    }

    @Test
    public void aSubgraphOfDegreeOneContainsDegreeOneConnections() {
        Set<IVertex> expectedDegree1Subgraph = ImmutableSet.of(v, parent, child);
        Set<IVertex> degree1SubGraph = network.getSubgraph(v, 1);
        assertThat(degree1SubGraph, containsInAnyOrder(expectedDegree1Subgraph.toArray()));
    }

    @Test
    public void aSubgraphOfDegreeTwoContainsDegreeOneAndTwoConnections() {
        Set<IVertex> expectedDegree2Subgraph = ImmutableSet.of(v, parent, child, grandChild, grandParent);
        Set<IVertex> degree2SubGraph = network.getSubgraph(v, 2);
        assertThat(degree2SubGraph, containsInAnyOrder(expectedDegree2Subgraph.toArray()));
    }

    @Test
    public void getSubgraphDegreeInfinityReturnsTheEntireGraph() {
        Set<IVertex> degreeInfinitySubgraph = network.getSubgraph(v, Integer.MAX_VALUE);
        List<IVertex> allVertices = network.getAllVertices();
        assertThat(degreeInfinitySubgraph, containsInAnyOrder(allVertices.toArray()));
    }
}
