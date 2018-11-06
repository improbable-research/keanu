package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex;
import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class ObservableTest {

    @Test
    public void youCanObserveANonProbabilisticBooleanVertex() {
        BoolVertex vertex = ConstantVertexFactory.of(true);
        assertFalse(vertex instanceof Probabilistic);
        vertex.observe(BooleanTensor.scalar(false));
    }

    @Test(expected = UnsupportedOperationException.class)
    public void youCannotObserveANonProbabilisticIntegerVertex() {
        IntegerVertex vertex = ConstantVertexFactory.of(1);
        assertFalse(vertex instanceof Probabilistic);
        vertex.observe(IntegerTensor.scalar(0));
    }

    @Test(expected = UnsupportedOperationException.class)
    public void youCannotObserveANonProbabilisticDoubleVertex() {
        DoubleVertex vertex = ConstantVertexFactory.of(1.0);
        assertFalse(vertex instanceof Probabilistic);
        vertex.observe(DoubleTensor.scalar(1.0));
    }

    @Test
    public void youCanObserveAProbabilisticBooleanVertex() {
        BoolVertex vertex = new BernoulliVertex(0.5);
        assertTrue(vertex instanceof Probabilistic);
        vertex.observe(BooleanTensor.scalar(false));
    }

    @Test
    public void youCanObserveAProbabilisticIntegerVertex() {
        IntegerVertex vertex = new UniformIntVertex(1, 1);
        assertTrue(vertex instanceof Probabilistic);
        vertex.observe(IntegerTensor.scalar(0));
    }

    @Test
    public void youCanObserveAProbabilisticDoubleVertex() {
        DoubleVertex vertex = new UniformVertex(1.0, 1.0);
        assertTrue(vertex instanceof Probabilistic);
        vertex.observe(DoubleTensor.scalar(1.0));
    }
}
