package io.improbable.keanu.vertices.bool;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.BaseBooleanTensor;
import io.improbable.keanu.kotlin.BooleanOperators;
import io.improbable.keanu.network.NetworkLoader;
import io.improbable.keanu.network.NetworkSaver;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.TensorVertex;
import io.improbable.keanu.vertices.Vertex;
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

public abstract class BooleanVertex extends Vertex<BooleanTensor>
    implements BooleanOperators<BooleanVertex>, TensorVertex<Boolean, BooleanVertex>, BaseBooleanTensor<BooleanVertex, IntegerVertex, DoubleVertex> {

    public BooleanVertex(long[] initialShape) {
        super(initialShape);
    }

    @Override
    public void saveValue(NetworkSaver netSaver) {
        netSaver.saveValue(this);
    }

    @Override
    public void loadValue(NetworkLoader loader) {
        loader.loadValue(this);
    }

    public void setValue(boolean value) {
        super.setValue(BooleanTensor.scalar(value));
    }

    public void setValue(boolean[] values) {
        super.setValue(BooleanTensor.create(values));
    }

    public void setAndCascade(boolean value) {
        super.setAndCascade(BooleanTensor.scalar(value));
    }

    public void setAndCascade(boolean[] values) {
        super.setAndCascade(BooleanTensor.create(values));
    }

    public void observe(boolean value) {
        super.observe(BooleanTensor.scalar(value));
    }

    public void observe(boolean[] values) {
        super.observe(BooleanTensor.create(values));
    }

    public boolean getValue(long... index) {
        return getValue().getValue(index);
    }

    public static BooleanVertex concat(int dimension, BooleanVertex... toConcat) {
        return new BooleanConcatenationVertex(dimension, toConcat);
    }

    public AssertVertex assertTrue() {
        return new AssertVertex(this);
    }

    public AssertVertex assertTrue(String errorMessage) {
        return new AssertVertex(this, errorMessage);
    }

    public BooleanVertex equalTo(BooleanVertex rhs) {
        return new EqualsVertex<>(this, rhs);
    }

    public <T extends Tensor> BooleanVertex notEqualTo(Vertex<T> rhs) {
        return new NotEqualsVertex<>(this, rhs);
    }

    @SafeVarargs
    public final BooleanVertex or(Vertex<BooleanTensor>... those) {
        if (those.length == 0) return this;
        if (those.length == 1) return new OrBinaryVertex(this, those[0]);
        return new OrMultipleVertex(inputList(those));
    }

    @Override
    public BooleanVertex or(boolean that) {
        return this.or(new ConstantBooleanVertex(that));
    }

    @Override
    public BooleanVertex xor(BooleanVertex that) {
        return new XorBinaryVertex(this, that);
    }

    @Override
    public BooleanVertex or(BooleanVertex that) {
        return new OrBinaryVertex(this, that);
    }

    @SafeVarargs
    public final BooleanVertex and(Vertex<BooleanTensor>... those) {
        if (those.length == 0) return this;
        if (those.length == 1) return new AndBinaryVertex(this, those[0]);
        return new AndMultipleVertex(inputList(those));
    }

    @Override
    public BooleanVertex and(BooleanVertex that) {
        return new AndBinaryVertex(this, that);
    }

    @Override
    public BooleanVertex and(boolean that) {
        return this.and(new ConstantBooleanVertex(that));
    }

    private List<Vertex<BooleanTensor>> inputList(Vertex<BooleanTensor>[] those) {
        return ImmutableList.<Vertex<BooleanTensor>>builder()
            .addAll(Arrays.asList(those))
            .add(this)
            .build();
    }

    @Override
    public BooleanVertex not() {
        return BooleanVertex.not(this);
    }

    @Override
    public DoubleVertex doubleWhere(DoubleVertex trueValue, DoubleVertex falseValue) {
        return null;
    }

    @Override
    public IntegerVertex integerWhere(IntegerVertex trueValue, IntegerVertex falseValue) {
        return null;
    }

    @Override
    public BooleanVertex booleanWhere(BooleanVertex trueValue, BooleanVertex falseValue) {
        return null;
    }

    @Override
    public <T, TENSOR extends Tensor<T, TENSOR>> TENSOR where(TENSOR trueValue, TENSOR falseValue) {
        return null;
    }

    @Override
    public BooleanVertex allTrue() {
        return null;
    }

    @Override
    public BooleanVertex allFalse() {
        return null;
    }

    @Override
    public BooleanVertex anyTrue() {
        return null;
    }

    @Override
    public BooleanVertex anyFalse() {
        return null;
    }

    @Override
    public DoubleVertex toDoubleMask() {
        return null;
    }

    @Override
    public IntegerVertex toIntegerMask() {
        return null;
    }

    public static BooleanVertex not(Vertex<BooleanTensor> vertex) {
        return new NotBinaryVertex(vertex);
    }


    public BooleanVertex take(long... index) {
        return new BooleanTakeVertex(this, index);
    }

    @Override
    public List<BooleanVertex> split(int dimension, long... splitAtIndices) {
        return null;
    }

    @Override
    public BooleanVertex diag() {
        return null;
    }

    public BooleanVertex reshape(long... proposedShape) {
        return new BooleanReshapeVertex(this, proposedShape);
    }

    @Override
    public BooleanVertex permute(int... rearrange) {
        return null;
    }

    @Override
    public BooleanVertex broadcast(long... toShape) {
        return null;
    }

    @Override
    public BooleanVertex elementwiseEquals(BooleanVertex that) {
        return null;
    }

    @Override
    public BooleanVertex elementwiseEquals(Boolean value) {
        return elementwiseEquals(ConstantVertex.of(value));
    }

    @Override
    public BooleanVertex get(BooleanVertex booleanIndex) {
        return null;
    }

    public BooleanVertex slice(int dimension, long index) {
        return new BooleanSliceVertex(this, dimension, index);
    }

    @Override
    public BooleanVertex slice(Slicer slicer) {
        return null;
    }
}
