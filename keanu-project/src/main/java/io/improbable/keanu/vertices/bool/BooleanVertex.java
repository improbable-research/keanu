package io.improbable.keanu.vertices.bool;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.BaseBooleanTensor;
import io.improbable.keanu.kotlin.BooleanOperators;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.AndBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.OrBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.XorBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.AndMultipleVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.BooleanConcatenationVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.BooleanToDoubleMaskVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.BooleanToIntegerMaskVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.OrMultipleVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.AllFalseVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.AllTrueVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.AnyFalseVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.AnyTrueVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.NotBinaryVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.utility.AssertVertex;

import java.util.Arrays;
import java.util.List;

public interface BooleanVertex extends
    BooleanOperators<BooleanVertex>,
    TensorVertex<Boolean, BooleanTensor, BooleanVertex>,
    BaseBooleanTensor<BooleanVertex, IntegerVertex, DoubleVertex> {

    /////////////
    //// Vertex
    /////////////

    default void setValue(boolean value) {
        setValue(BooleanTensor.scalar(value));
    }

    default void setValue(boolean[] values) {
        setValue(BooleanTensor.create(values));
    }

    default void setAndCascade(boolean value) {
        setAndCascade(BooleanTensor.scalar(value));
    }

    default void setAndCascade(boolean[] values) {
        setAndCascade(BooleanTensor.create(values));
    }

    default void observe(boolean value) {
        observe(BooleanTensor.scalar(value));
    }

    default void observe(boolean[] values) {
        observe(BooleanTensor.create(values));
    }

    default boolean getValue(long... index) {
        return getValue().getValue(index);
    }

    @Override
    default Class<?> ofType() {
        return BooleanTensor.class;
    }

    @Override
    default BooleanVertex wrap(NonProbabilisticVertex<BooleanTensor, BooleanVertex> vertex) {
        return new BooleanVertexWrapper(vertex);
    }

    /////////////
    //// Tensor Ops
    /////////////

    static BooleanVertex concat(int dimension, BooleanVertex... toConcat) {
        return new BooleanConcatenationVertex(dimension, toConcat);
    }

    default AssertVertex assertTrue() {
        return new AssertVertex(this);
    }

    default AssertVertex assertTrue(String errorMessage) {
        return new AssertVertex(this, errorMessage);
    }

    /////////////
    //// Boolean Ops
    /////////////

    default BooleanVertex or(BooleanVertex... those) {
        if (those.length == 0) return this;
        if (those.length == 1) return new OrBinaryVertex(this, those[0]);
        List<BooleanVertex> list = ImmutableList.<BooleanVertex>builder()
            .addAll(Arrays.asList(those))
            .add(this)
            .build();
        return new OrMultipleVertex(list);
    }

    @Override
    default BooleanVertex or(boolean that) {
        return this.or(new ConstantBooleanVertex(that));
    }

    @Override
    default BooleanVertex or(BooleanVertex that) {
        return new OrBinaryVertex(this, that);
    }

    default BooleanVertex and(BooleanVertex... those) {
        if (those.length == 0) return this;
        if (those.length == 1) return new AndBinaryVertex(this, those[0]);
        List<BooleanVertex> list = ImmutableList.<BooleanVertex>builder()
            .addAll(Arrays.asList(those))
            .add(this)
            .build();
        return new AndMultipleVertex(list);
    }

    @Override
    default BooleanVertex and(BooleanVertex that) {
        return new AndBinaryVertex(this, that);
    }

    @Override
    default BooleanVertex and(boolean that) {
        return this.and(new ConstantBooleanVertex(that));
    }

    @Override
    default BooleanVertex xor(BooleanVertex that) {
        return new XorBinaryVertex(this, that);
    }

    @Override
    default BooleanVertex not() {
        return BooleanVertex.not(this);
    }

    static BooleanVertex not(BooleanVertex vertex) {
        return new NotBinaryVertex(vertex);
    }

    @Override
    default BooleanVertex allTrue() {
        return new AllTrueVertex(this);
    }

    @Override
    default BooleanVertex allFalse() {
        return new AllFalseVertex(this);
    }

    @Override
    default BooleanVertex anyTrue() {
        return new AnyTrueVertex(this);
    }

    @Override
    default BooleanVertex anyFalse() {
        return new AnyFalseVertex(this);
    }

    @Override
    default DoubleVertex toDoubleMask() {
        return new BooleanToDoubleMaskVertex(this);
    }

    @Override
    default IntegerVertex toIntegerMask() {
        return new BooleanToIntegerMaskVertex(this);
    }


}
