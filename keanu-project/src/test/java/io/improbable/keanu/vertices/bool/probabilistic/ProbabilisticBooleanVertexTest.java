package io.improbable.keanu.vertices.bool.probabilistic;

import static org.hamcrest.MatcherAssert.assertThat;

import static io.improbable.keanu.tensor.TensorMatchers.tensorEqualTo;

import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.junit.Test;

import com.google.common.primitives.Booleans;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class ProbabilisticBooleanVertexTest {
    @Test
    public void sampleUsesShapeOfTensor() {
        ProbabilisticBooleanVertex vertex1 = new ProbabilisticBooleanVertexImplementation(1, 1);

        assertThat(vertex1.sample(), tensorEqualTo(BooleanTensor.scalar(true)));

        ProbabilisticBooleanVertex vertex2 = new ProbabilisticBooleanVertexImplementation(2, 3);

        assertThat(vertex2.sample(), tensorEqualTo(BooleanTensor.create(true, false, true, false, true, false).reshape(2, 3)));
    }

    @Test
    public void sampleManyScalarsUsesInputShape() {
        ProbabilisticBooleanVertex vertex = new ProbabilisticBooleanVertexImplementation(1, 1);

        assertThat(vertex.sampleManyScalars(new long[]{1, 1}), tensorEqualTo(BooleanTensor.scalar(true)));
        assertThat(vertex.sampleManyScalars(new long[]{2, 3, 2}), tensorEqualTo(BooleanTensor.create(true, false, true, false, true, false, true, false, true, false, true, false).reshape(2, 3, 2)));
    }

    @Test(expected = IllegalArgumentException.class)
    public void cannotSampleManyScalarsOfNonScalar() {
        ProbabilisticBooleanVertex vertex = new ProbabilisticBooleanVertexImplementation(2, 2);

        vertex.sampleManyScalars(new long[]{1, 1});
    }

    private static class ProbabilisticBooleanVertexImplementation extends ProbabilisticBooleanVertex {
        ProbabilisticBooleanVertexImplementation(long... shape) {
            this.setValue(BooleanTensor.placeHolder(shape));
        }

        @Override
        protected BooleanTensor sampleWithShape(long[] shape, KeanuRandom random) {
            int length = Math.toIntExact(TensorShape.getLength(shape));
            Stream<Boolean> trueFalseStream = IntStream.range(0, length).mapToObj(i -> i % 2 == 0);
            return BooleanTensor.create(Booleans.toArray(trueFalseStream.collect(Collectors.toList()))).reshape(shape);
        }

        @Override
        public double logProb(BooleanTensor value) {
            return 0;
        }

        @Override
        public Map<Vertex, DoubleTensor> dLogProb(BooleanTensor atValue, Set<? extends Vertex> withRespectTo) {
            return null;
        }
    }
}
