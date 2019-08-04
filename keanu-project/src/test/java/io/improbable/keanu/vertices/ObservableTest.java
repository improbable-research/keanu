package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.tensor.bool.BooleanVertex;
import io.improbable.keanu.vertices.tensor.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic.UniformIntVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class ObservableTest {

    @Test
    public void youCanObserveANonProbabilisticBooleanVertex() {
        BooleanVertex vertex = ConstantVertex.of(true);
        assertFalse(vertex instanceof Probabilistic);
        vertex.observe(BooleanTensor.scalar(false));
    }

    @Test(expected = UnsupportedOperationException.class)
    public void youCannotObserveANonProbabilisticIntegerVertex() {
        IntegerVertex vertex = ConstantVertex.of(1);
        assertFalse(vertex instanceof Probabilistic);
        vertex.observe(IntegerTensor.scalar(0));
    }

    @Test(expected = UnsupportedOperationException.class)
    public void youCannotObserveANonProbabilisticDoubleVertex() {
        DoubleVertex vertex = ConstantVertex.of(1.0);
        assertFalse(vertex instanceof Probabilistic);
        vertex.observe(DoubleTensor.scalar(1.0));
    }

    @Test
    public void youCanObserveAProbabilisticBooleanVertex() {
        BooleanVertex vertex = new BernoulliVertex(0.5);
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
