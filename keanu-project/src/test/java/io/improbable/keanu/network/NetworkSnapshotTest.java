package io.improbable.keanu.network;

import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexState;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.is;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doCallRealMethod;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

public class NetworkSnapshotTest {

    @Test
    public void itInspectsTheVerticesState() {
        Vertex v1 = mock(Vertex.class);
        Vertex v2 = mock(Vertex.class);

        NetworkSnapshot.create(ImmutableSet.of(v1, v2));
        verify(v1).getState();
        verify(v2).getState();
        verifyNoMoreInteractions(v1, v2);
    }

    @Test
    public void itSetsTheStateOfAVertex() {
        VertexState<Object> s1 = VertexState.nullState();
        VertexState<Object> s2 = VertexState.nullState();
        Vertex<Object> v1 = mock(Vertex.class);
        Vertex<Object> v2 = mock(Vertex.class);
        when(v1.getState()).thenReturn(s1);
        when(v2.getState()).thenReturn(s2);
        doCallRealMethod().when(v1).setState(any(VariableState.class));
        doCallRealMethod().when(v2).setState(any(VariableState.class));

        NetworkSnapshot snapshot = NetworkSnapshot.create(ImmutableSet.of(v1, v2));
        verify(v1).getState();
        verify(v2).getState();

        snapshot.apply();
        verify(v1).setState(s1);
        verify(v2).setState(s2);
    }

    @Test
    public void itRestoresTheValueOfAnUnobservedVertex() {
        DoubleTensor originalValue = DoubleTensor.create(1., 2., 3.);
        DoubleTensor otherValue = DoubleTensor.create(4., 5., 6.);
        Vertex vertex = new GaussianVertex(1., 0.);
        vertex.setValue(originalValue);
        NetworkSnapshot snapshot = NetworkSnapshot.create(ImmutableSet.of(vertex));

        vertex.setValue(otherValue);
        assertThat(vertex.getValue(), equalTo(otherValue));
        assertThat(vertex.isObserved(), is(false));

        snapshot.apply();
        assertThat(vertex.getValue(), equalTo(originalValue));
        assertThat(vertex.isObserved(), is(false));
    }

    @Test
    public void itRestoresTheValueOfAnObservedVertex() {
        DoubleTensor originalValue = DoubleTensor.create(1., 2., 3.);
        DoubleTensor otherValue = DoubleTensor.create(4., 5., 6.);
        Vertex vertex = new GaussianVertex(1., 0.);

        vertex.observe(originalValue);
        assertThat(vertex.getValue(), equalTo(originalValue));
        assertThat(vertex.isObserved(), is(true));

        NetworkSnapshot snapshot = NetworkSnapshot.create(ImmutableSet.of(vertex));

        vertex.observe(otherValue);
        assertThat(vertex.getValue(), equalTo(otherValue));
        assertThat(vertex.isObserved(), is(true));

        snapshot.apply();
        assertThat(vertex.getValue(), equalTo(originalValue));
        assertThat(vertex.isObserved(), is(true));
    }

    @Test
    public void itRestoresTheObservedStatusOfAnObservedVertex() {
        DoubleTensor originalValue = DoubleTensor.create(1., 2., 3.);
        DoubleTensor otherValue = DoubleTensor.create(4., 5., 6.);
        Vertex vertex = new GaussianVertex(1., 0.);

        vertex.observe(originalValue);
        assertThat(vertex.getValue(), equalTo(originalValue));
        assertThat(vertex.isObserved(), is(true));

        NetworkSnapshot snapshot = NetworkSnapshot.create(ImmutableSet.of(vertex));

        vertex.unobserve();
        assertThat(vertex.getValue(), equalTo(originalValue));
        assertThat(vertex.isObserved(), is(false));

        vertex.setValue(otherValue);
        assertThat(vertex.getValue(), equalTo(otherValue));
        assertThat(vertex.isObserved(), is(false));

        snapshot.apply();
        assertThat(vertex.getValue(), equalTo(originalValue));
        assertThat(vertex.isObserved(), is(true));
    }
}