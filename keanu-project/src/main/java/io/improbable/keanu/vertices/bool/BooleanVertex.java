package io.improbable.keanu.vertices.bool;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.kotlin.BooleanOperators;
import io.improbable.keanu.network.NetworkLoader;
import io.improbable.keanu.network.NetworkSaver;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.AndBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.OrBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.EqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NotEqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.AndMultipleVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.BooleanConcatenationVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.OrMultipleVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BooleanReshapeVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BooleanSliceVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BooleanTakeVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.NotVertex;

import java.util.Arrays;
import java.util.List;

public abstract class BooleanVertex extends Vertex<BooleanTensor> implements BooleanOperators<BooleanVertex> {

    public BooleanVertex(long[] initialShape) {
        super(initialShape);
    }

    public void saveValue(NetworkSaver netSaver) {
        netSaver.saveValue(this);
    }

    @Override
    public void loadValue(NetworkLoader loader) {
        loader.loadValue(this);
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

    @Override
    public BooleanVertex not() {
        return BooleanVertex.not(this);
    }

    public static BooleanVertex concat(int dimension, BooleanVertex... toConcat) {
        return new BooleanConcatenationVertex(dimension, toConcat);
    }

    public static final BooleanVertex not(Vertex<BooleanTensor> vertex) {
        return new NotVertex(vertex);
    }

    public BooleanVertex equalTo(BooleanVertex rhs) {
        return new EqualsVertex<>(this, rhs);
    }

    public <T extends Tensor> BooleanVertex notEqualTo(Vertex<T> rhs) {
        return new NotEqualsVertex<>(this, rhs);
    }

    private List<Vertex<BooleanTensor>> inputList(Vertex<BooleanTensor>[] those) {
        return ImmutableList.<Vertex<BooleanTensor>>builder()
            .addAll(Arrays.asList(those))
            .add(this)
            .build();
    }

    public BooleanVertex slice(int dimension, long index) {
        return new BooleanSliceVertex(this, dimension, index);
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

    public BooleanVertex take(long... index) {
        return new BooleanTakeVertex(this, index);
    }

    public BooleanVertex reshape(long... proposedShape) {
        return new BooleanReshapeVertex(this, proposedShape);
    }


}
