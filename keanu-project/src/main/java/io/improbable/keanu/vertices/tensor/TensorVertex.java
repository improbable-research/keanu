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

public interface TensorVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends Vertex<TENSOR, VERTEX>, BaseTensor<BooleanVertex, T, VERTEX> {

    VERTEX wrap(NonProbabilisticVertex<TENSOR, VERTEX> vertex);

    @Override
    default VERTEX get(BooleanVertex booleanIndex) {
        return wrap(new GetBooleanIndexVertex<>(this, booleanIndex));
    }

    @Override
    default VERTEX slice(int dimension, long index) {
        return wrap(new SliceVertex<>(this, dimension, index));
    }

    @Override
    default VERTEX slice(Slicer slicer) {
        return wrap(new StridedSliceVertex<>(this, slicer));
    }

    @Override
    default VERTEX diag() {
        return wrap(new DiagVertex<>(this));
    }

    @Override
    default VERTEX diagPart() {
        return wrap(new DiagPartVertex<>(this));
    }

    @Override
    default VERTEX where(BooleanVertex predicate, VERTEX els) {
        return wrap(new WhereVertex<>(predicate, this, els));
    }

    @Override
    default VERTEX take(long... index) {
        return wrap(new TakeVertex<>(this, index));
    }

    @Override
    default VERTEX reshape(long... proposedShape) {
        return wrap(new ReshapeVertex<>(this, proposedShape));
    }

    @Override
    default VERTEX permute(int... rearrange) {
        return wrap(new PermuteVertex<>(this, rearrange));
    }

    @Override
    default VERTEX broadcast(long... toShape) {
        return wrap(new BroadcastVertex<>(this, toShape));
    }

    @Override
    default BooleanVertex elementwiseEquals(VERTEX that) {
        return new EqualsVertex<>(this, that);
    }

    @Override
    default BooleanVertex notEqualTo(VERTEX that) {
        return new NotEqualsVertex<>(this, that);
    }

    @Override
    default BooleanVertex elementwiseEquals(T that) {
        return elementwiseEquals((VERTEX) ConstantVertex.scalar(that));
    }

    @Override
    default BooleanVertex notEqualTo(T that) {
        return notEqualTo(((VERTEX) ConstantVertex.scalar(that)));
    }

}
