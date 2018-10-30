package io.improbable.keanu.vertices.bool;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.kotlin.BooleanOperators;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.AndBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.OrBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.EqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NotEqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.AndMultipleVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.BoolConcatenationVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.OrMultipleVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BoolReshapeVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BoolSliceVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BoolTakeVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.NotVertex;

import java.util.Arrays;
import java.util.List;

public abstract class BoolVertex extends Vertex<BooleanTensor> implements BooleanOperators<BoolVertex> {

    public BoolVertex(long[] initialShape) {
        super(initialShape);
    }

    @SafeVarargs
    public final BoolVertex or(Vertex<BooleanTensor>... those) {
        if (those.length == 0) return this;
        if (those.length == 1) return new OrBinaryVertex(this, those[0]);
        return new OrMultipleVertex(inputList(those));
    }

    @Override
    public BoolVertex or(boolean that) {
        return this.or(new ConstantBoolVertex(that));
    }

    @Override
    public BoolVertex or(BoolVertex that) {
        return new OrBinaryVertex(this, that);
    }

    @SafeVarargs
    public final BoolVertex and(Vertex<BooleanTensor>... those) {
        if (those.length == 0) return this;
        if (those.length == 1) return new AndBinaryVertex(this, those[0]);
        return new AndMultipleVertex(inputList(those));
    }

    @Override
    public BoolVertex and(BoolVertex that) {
        return new AndBinaryVertex(this, that);
    }

    @Override
    public BoolVertex and(boolean that) {
        return this.and(new ConstantBoolVertex(that));
    }

    @Override
    public BoolVertex not() {
        return BoolVertex.not(this);
    }

    public static BoolVertex concat(int dimension, BoolVertex... toConcat) {
        return new BoolConcatenationVertex(dimension, toConcat);
    }

    public static final BoolVertex not(Vertex<BooleanTensor> vertex) {
        return new NotVertex(vertex);
    }

    public BoolVertex equalTo(BoolVertex rhs) {
        return new EqualsVertex<>(this, rhs);
    }

    public <T extends Tensor> BoolVertex notEqualTo(Vertex<T> rhs) {
        return new NotEqualsVertex<>(this, rhs);
    }

    private List<Vertex<BooleanTensor>> inputList(Vertex<BooleanTensor>[] those) {
        return ImmutableList.<Vertex<BooleanTensor>>builder()
            .addAll(Arrays.asList(those))
            .add(this)
            .build();
    }

    public BoolVertex slice(int dimension, long index) {
        return new BoolSliceVertex(this, dimension, index);
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

    public BoolVertex take(long... index) {
        return new BoolTakeVertex(this, index);
    }

    public BoolVertex reshape(long... proposedShape) {
        return new BoolReshapeVertex(this, proposedShape);
    }


}
