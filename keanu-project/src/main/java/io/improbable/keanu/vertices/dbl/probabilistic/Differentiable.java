package io.improbable.keanu.vertices.dbl.probabilistic;

import java.util.List;
import java.util.Map;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public interface Differentiable {

    DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers);

    default DualNumber getDualNumber() {
        return new Differentiator().calculateDual((Vertex & Differentiable) this);
    }

    static <V extends Vertex & Differentiable> List<V> filter(List<? extends Vertex<?>> vertices) {
        ImmutableList.Builder<V> differentiableVertices = ImmutableList.builder();
        for (Vertex v : vertices) {
            if (v instanceof Differentiable) {
                differentiableVertices.add((V) v);
            }
        }
        return differentiableVertices.build();
    }
}
