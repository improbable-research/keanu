package io.improbable.keanu.network;

import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.is;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

public class NetworkSnapshotTest {

    @Test
    public void itInspectsTheVerticesValueAndObservedStatus() {
        Vertex v1 = mock(Vertex.class);
        when(v1.getValue()).thenReturn(DoubleTensor.create(1., 2., 3.));
        when(v1.isObserved()).thenReturn(false);
        Vertex v2 = mock(Vertex.class);
        when(v2.getValue()).thenReturn(DoubleTensor.create(11., 12., 13., 14.));
        when(v2.isObserved()).thenReturn(true);

        NetworkSnapshot.create(ImmutableSet.of(v1, v2));
        verify(v1).getValue();
        verify(v2).getValue();
        verify(v1).isObserved();
        verify(v2).isObserved();
        verifyNoMoreInteractions(v1, v2);
    }

    @Test
    public void itSetsTheStateOfAnUnobservedVertex() {
        Vertex vertex = mock(Vertex.class);
        DoubleTensor value = DoubleTensor.create(1., 2., 3.);
        when(vertex.getValue()).thenReturn(value);
        when(vertex.isObserved()).thenReturn(false);

        NetworkSnapshot snapshot = NetworkSnapshot.create(ImmutableSet.of(vertex));
        snapshot.apply();
        verify(vertex).unobserve();
        verify(vertex).setValue(value);
    }

    @Test
    public void itSetsTheStateOfAnObservedVertex() {
        Vertex vertex = mock(Vertex.class);
        DoubleTensor value = DoubleTensor.create(1., 2., 3.);
        when(vertex.getValue()).thenReturn(value);
        when(vertex.isObserved()).thenReturn(true);

        NetworkSnapshot snapshot = NetworkSnapshot.create(ImmutableSet.of(vertex));
        snapshot.apply();
        verify(vertex).observe(value);
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