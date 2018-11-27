package io.improbable.keanu.network;

import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.vertices.Vertex;
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

    private static Vertex v;
    private static BayesianNetwork network;
    private static Vertex grandChild;
    private static Vertex child;
    private static Vertex parent;
    private static Vertex grandParent;

    @BeforeClass
    public static void setUp() {
        v = Mockito.mock(Vertex.class);

        Set<Vertex> parents = new HashSet<>();
        parent = Mockito.mock(Vertex.class);
        parents.add(parent);
        Set<Vertex> children = new HashSet<>();
        child = Mockito.mock(Vertex.class);
        children.add(child);
        when(v.getParents()).thenReturn(parents);
        when(v.getChildren()).thenReturn(children);

        Set<Vertex> grandParents = new HashSet<>();
        grandParent = Mockito.mock(Vertex.class);
        grandParents.add(grandParent);
        Set<Vertex> grandChildren = new HashSet<>();
        grandChild = Mockito.mock(Vertex.class);
        grandChildren.add(grandChild);
        when(parent.getParents()).thenReturn(grandParents);
        when(child.getChildren()).thenReturn(grandChildren);

        when(v.getConnectedGraph()).thenReturn(ImmutableSet.of(v, parent, child, grandChild, grandParent));

        network = new BayesianNetwork(v.getConnectedGraph());
    }

    @Test
    public void aSubgraphOfDegreeZeroContainsOneVertex() {
        Set<Vertex> degree0SubGraph = network.getSubgraph(v, 0);
        assertThat(degree0SubGraph, contains(v));
    }

    @Test
    public void aSubgraphOfDegreeOneContainsDegreeOneConnections() {
        Set<Vertex> expectedDegree1Subgraph = ImmutableSet.of(v, parent, child);
        Set<Vertex> degree1SubGraph = network.getSubgraph(v, 1);
        assertThat(degree1SubGraph, containsInAnyOrder(expectedDegree1Subgraph.toArray()));
    }

    @Test
    public void aSubgraphOfDegreeTwoContainsDegreeOneAndTwoConnections() {
        Set<Vertex> expectedDegree2Subgraph = ImmutableSet.of(v, parent, child, grandChild, grandParent);
        Set<Vertex> degree2SubGraph = network.getSubgraph(v, 2);
        assertThat(degree2SubGraph, containsInAnyOrder(expectedDegree2Subgraph.toArray()));
    }

    @Test
    public void getSubgraphDegreeInfinityReturnsTheEntireGraph() {
        Set<Vertex> degreeInfinitySubgraph = network.getSubgraph(v, Integer.MAX_VALUE);
        List<Vertex> allVertices = network.getAllVertices();
        assertThat(degreeInfinitySubgraph, containsInAnyOrder(allVertices.toArray()));
    }
}
