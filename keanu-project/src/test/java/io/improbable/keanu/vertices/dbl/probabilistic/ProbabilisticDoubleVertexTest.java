package io.improbable.keanu.vertices.dbl.probabilistic;

import static org.hamcrest.MatcherAssert.assertThat;

import static io.improbable.keanu.tensor.TensorMatchers.tensorEqualTo;

import java.util.Map;
import java.util.Set;

import org.junit.Test;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class ProbabilisticDoubleVertexTest {
    @Test
    public void sampleUsesShapeOfTensor() {
        ProbabilisticDoubleVertex vertex1 = new ProbabilisticDoubleVertexImplementation(1, 1);

        assertThat(vertex1.sample(), tensorEqualTo(DoubleTensor.scalar(0)));

        ProbabilisticDoubleVertex vertex2 = new ProbabilisticDoubleVertexImplementation(2, 3);

        assertThat(vertex2.sample(), tensorEqualTo(DoubleTensor.create(0, 1, 2, 3, 4, 5).reshape(2, 3)));
    }

    @Test
    public void sampleManyScalarsUsesInputShape() {
        ProbabilisticDoubleVertex vertex = new ProbabilisticDoubleVertexImplementation(1, 1);

        assertThat(vertex.sampleManyScalars(new long[]{1, 1}), tensorEqualTo(DoubleTensor.scalar(0)));
        assertThat(vertex.sampleManyScalars(new long[]{2, 3, 2}), tensorEqualTo(DoubleTensor.create(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11).reshape(2, 3, 2)));
    }

    @Test(expected = IllegalArgumentException.class)
    public void cannotSampleManyScalarsOfNonScalar() {
        ProbabilisticDoubleVertex vertex = new ProbabilisticDoubleVertexImplementation(2, 2);

        vertex.sampleManyScalars(new long[]{1, 1});
    }

    private static class ProbabilisticDoubleVertexImplementation extends ProbabilisticDoubleVertex {
        ProbabilisticDoubleVertexImplementation(long... shape) {
            this.setValue(DoubleTensor.placeHolder(shape));
        }

        @Override
        protected DoubleTensor sampleWithShape(long[] shape, KeanuRandom random) {
            int length = Math.toIntExact(TensorShape.getLength(shape));
            return DoubleTensor.arange(0, length).reshape(shape);
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
}
