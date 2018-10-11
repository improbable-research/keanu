package io.improbable.keanu.vertices.intgr.probabilistic;


import static org.hamcrest.MatcherAssert.assertThat;

import static io.improbable.keanu.tensor.TensorMatchers.tensorEqualTo;

import java.util.Map;
import java.util.Set;
import java.util.stream.IntStream;

import org.junit.Test;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class ProbabilisticIntegerVertexTest {

    @Test
    public void sampleUsesShapeOfTensor() {
        ProbabilisticIntegerVertex vertex1 = new ProbabilisticIntegerVertexImplementation(1, 1);

        assertThat(vertex1.sample(), tensorEqualTo(IntegerTensor.scalar(0)));

        ProbabilisticIntegerVertex vertex2 = new ProbabilisticIntegerVertexImplementation(2, 3);

        assertThat(vertex2.sample(), tensorEqualTo(IntegerTensor.create(0, 1, 2, 3, 4, 5).reshape(2, 3)));
    }

    @Test
    public void sampleManyScalarsUsesInputShape() {
        ProbabilisticIntegerVertex vertex = new ProbabilisticIntegerVertexImplementation(1, 1);

        assertThat(vertex.sampleManyScalars(new long[]{1, 1}), tensorEqualTo(IntegerTensor.scalar(0)));
        assertThat(vertex.sampleManyScalars(new long[]{2, 3, 2}), tensorEqualTo(IntegerTensor.create(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11).reshape(2, 3, 2)));
    }

    @Test(expected = IllegalArgumentException.class)
    public void cannotSampleManyScalarsOfNonScalar() {
        ProbabilisticIntegerVertex vertex = new ProbabilisticIntegerVertexImplementation(2, 2);

        vertex.sampleManyScalars(new long[]{1, 1});
    }

    private static class ProbabilisticIntegerVertexImplementation extends ProbabilisticIntegerVertex {
        ProbabilisticIntegerVertexImplementation(long... shape) {
            this.setValue(IntegerTensor.placeHolder(shape));
        }

        @Override
        protected IntegerTensor sampleWithShape(long[] shape, KeanuRandom random) {
            int length = Math.toIntExact(TensorShape.getLength(shape));
            return IntegerTensor.create(IntStream.range(0, length).toArray(), shape);
        }

        @Override
        public double logProb(IntegerTensor value) {
            return 0;
        }

        @Override
        public Map<Vertex, DoubleTensor> dLogProb(IntegerTensor atValue, Set<? extends Vertex> withRespectTo) {
            return null;
        }
    }
}
