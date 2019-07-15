package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.BaseTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.EqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NotEqualsVertex;

import java.util.List;

public interface TensorVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends Vertex<TENSOR, VERTEX>, BaseTensor<BooleanVertex, T, VERTEX> {

    VERTEX wrap(NonProbabilisticVertex<TENSOR, VERTEX> vertex);

    default VERTEX get(BooleanVertex booleanIndex) {
        return wrap(new GetBooleanIndexVertex<>(this, booleanIndex));
    }

    default VERTEX slice(int dimension, long index) {
        return wrap(new SliceVertex<>(this, dimension, index));
    }

    default VERTEX slice(Slicer slicer) {
        return null;
    }

    default List<VERTEX> split(int dimension, long... splitAtIndices) {
        return null;
    }

    default VERTEX diag() {
        return wrap(new DiagVertex<>(this));
    }

    default VERTEX take(long... index) {
        return wrap(new TakeVertex<>(this, index));
    }

    default VERTEX reshape(long... proposedShape) {
        return wrap(new ReshapeVertex<>(this, proposedShape));
    }

    default VERTEX permute(int... rearrange) {
        return wrap(new PermuteVertex<>(this, rearrange));
    }

    default VERTEX broadcast(long... toShape) {
        return wrap(new BroadcastVertex<>(this, toShape));
    }

    default BooleanVertex elementwiseEquals(VERTEX that) {
        return new EqualsVertex<>(this, that);
    }

    default BooleanVertex notEqualTo(VERTEX that) {
        return new NotEqualsVertex<>(this, that);
    }

    default BooleanVertex elementwiseEquals(T that) {
        return elementwiseEquals((VERTEX) ConstantVertex.scalar(that));
    }

    default BooleanVertex notEqualTo(T that) {
        return notEqualTo(((VERTEX) ConstantVertex.scalar(that)));
    }

}
