package io.improbable.keanu.vertices.bool;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.BaseBooleanTensor;
import io.improbable.keanu.kotlin.BooleanOperators;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.TensorVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.AndBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.OrBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.XorBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.EqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NotEqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.AndMultipleVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.BooleanConcatenationVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.OrMultipleVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BooleanReshapeVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BooleanSliceVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BooleanTakeVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.NotBinaryVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
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

    default BooleanVertex equalTo(BooleanVertex rhs) {
        return new EqualsVertex<>(this, rhs);
    }

    default <T extends Tensor> BooleanVertex notEqualTo(IVertex<T> rhs) {
        return new NotEqualsVertex<>(this, rhs);
    }

    default BooleanVertex take(long... index) {
        return new BooleanTakeVertex(this, index);
    }

    @Override
    default List<BooleanVertex> split(int dimension, long... splitAtIndices) {
        return null;
    }

    @Override
    default BooleanVertex diag() {
        return null;
    }

    default BooleanVertex reshape(long... proposedShape) {
        return new BooleanReshapeVertex(this, proposedShape);
    }

    @Override
    default BooleanVertex permute(int... rearrange) {
        return null;
    }

    @Override
    default BooleanVertex broadcast(long... toShape) {
        return null;
    }

    @Override
    default BooleanVertex elementwiseEquals(BooleanVertex that) {
        return null;
    }

    @Override
    default BooleanVertex elementwiseEquals(Boolean value) {
        return elementwiseEquals(ConstantVertex.of(value));
    }

    @Override
    default BooleanVertex get(BooleanVertex booleanIndex) {
        return null;
    }

    default BooleanVertex slice(int dimension, long index) {
        return new BooleanSliceVertex(this, dimension, index);
    }

    @Override
    default BooleanVertex slice(Slicer slicer) {
        return null;
    }

    /////////////
    //// Boolean Ops
    /////////////

    default BooleanVertex or(IVertex<BooleanTensor>... those) {
        if (those.length == 0) return this;
        if (those.length == 1) return new OrBinaryVertex(this, those[0]);
        List<IVertex<BooleanTensor>> list = ImmutableList.<IVertex<BooleanTensor>>builder()
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
    default BooleanVertex xor(BooleanVertex that) {
        return new XorBinaryVertex(this, that);
    }

    @Override
    default BooleanVertex or(BooleanVertex that) {
        return new OrBinaryVertex(this, that);
    }


    default BooleanVertex and(IVertex<BooleanTensor>... those) {
        if (those.length == 0) return this;
        if (those.length == 1) return new AndBinaryVertex(this, those[0]);
        List<IVertex<BooleanTensor>> list = ImmutableList.<IVertex<BooleanTensor>>builder()
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
    default BooleanVertex not() {
        return BooleanVertex.not(this);
    }

    @Override
    default DoubleVertex doubleWhere(DoubleVertex trueValue, DoubleVertex falseValue) {
        return null;
    }

    @Override
    default IntegerVertex integerWhere(IntegerVertex trueValue, IntegerVertex falseValue) {
        return null;
    }

    @Override
    default BooleanVertex booleanWhere(BooleanVertex trueValue, BooleanVertex falseValue) {
        return null;
    }

    @Override
    default <T, TENSOR extends Tensor<T, TENSOR>> TENSOR where(TENSOR trueValue, TENSOR falseValue) {
        return null;
    }

    @Override
    default BooleanVertex allTrue() {
        return null;
    }

    @Override
    default BooleanVertex allFalse() {
        return null;
    }

    @Override
    default BooleanVertex anyTrue() {
        return null;
    }

    @Override
    default BooleanVertex anyFalse() {
        return null;
    }

    @Override
    default DoubleVertex toDoubleMask() {
        return null;
    }

    @Override
    default IntegerVertex toIntegerMask() {
        return null;
    }

    static BooleanVertex not(IVertex<BooleanTensor> vertex) {
        return new NotBinaryVertex(vertex);
    }

}
