package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.BaseTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.EqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NotEqualsVertex;

public interface TensorVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends Vertex<TENSOR, VERTEX>, BaseTensor<BooleanVertex, T, VERTEX> {

    VERTEX asTyped(NonProbabilisticVertex<TENSOR, VERTEX> vertex);

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
