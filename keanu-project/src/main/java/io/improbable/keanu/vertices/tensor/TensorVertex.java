package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.BaseTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.EqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NotEqualsVertex;

import java.util.List;

public interface TensorVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends Vertex<TENSOR, VERTEX>, BaseTensor<BooleanVertex, T, VERTEX> {

    VERTEX asTyped(NonProbabilisticVertex<TENSOR, VERTEX> vertex);

    default VERTEX get(BooleanVertex booleanIndex) {
        return asTyped(new GetBooleanIndexVertex<>(this, booleanIndex));
    }

    default VERTEX slice(int dimension, long index) {
        return asTyped(new SliceVertex<>(this, dimension, index));
    }

    default VERTEX slice(Slicer slicer) {
        return null;
    }

    default List<VERTEX> split(int dimension, long... splitAtIndices) {
        return null;
    }

    default VERTEX diag() {
        return null;
    }

    default VERTEX take(long... index) {
        return asTyped(new TakeVertex<>(this, index));
    }

    default VERTEX reshape(long... proposedShape) {
        return asTyped(new ReshapeVertex<>(this, proposedShape));
    }

    default VERTEX permute(int... rearrange) {
        return asTyped(new PermuteVertex<>(this, rearrange));
    }

    default VERTEX broadcast(long... toShape) {
        return asTyped(new BroadcastVertex<>(this, toShape));
    }

    default BooleanVertex elementwiseEquals(VERTEX that) {
        return new EqualsVertex<>(this, that);
    }

    default BooleanVertex notEqualTo(VERTEX rhs) {
        return new NotEqualsVertex<>(this, rhs);
    }

}
