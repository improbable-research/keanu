package io.improbable.keanu.vertices;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public final class VertexSampler {
    private VertexSampler() {
    }

    public static <T> List<T> sampleManyScalarsFromTensorVertex(Vertex<? extends Tensor<T>> vertex, int sampleCount, KeanuRandom random) {
        if (!TensorShape.isScalar(vertex.getShape())) {
            throw new IllegalArgumentException("Vertex to sample must be scalar");
        }

        return VertexSampler.sampleManyFromVertex(vertex, sample -> sample.scalar(), sampleCount, random);
    }

    private static <I, O> List<O> sampleManyFromVertex(Vertex<I> vertex, Function<I, O> convertSample, int sampleCount, KeanuRandom random) {
        final ArrayList<O> samples = new ArrayList<>(sampleCount);

        for (int i = 0; i < sampleCount; i++) {
            samples.add(convertSample.apply(vertex.sample(random)));
        }

        return samples;
    }
}
