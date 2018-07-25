package io.improbable.keanu.vertices.bool;

import java.util.Arrays;
import java.util.List;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BooleanBinaryOpVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.AndMultipleVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.OrMultipleVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BooleanUnaryOpVertex;
import io.improbable.keanu.vertices.update.ValueUpdater;

public abstract class BooleanVertex extends Vertex<BooleanTensor> {

    public BooleanVertex(ValueUpdater<BooleanTensor> valueUpdater, Observable<BooleanTensor> observation) {
        super(valueUpdater, observation);
    }

    @SafeVarargs
    public final BooleanVertex or(Vertex<BooleanTensor>... those) {
        if (those.length == 0) return this;
        if (those.length == 1) return new BooleanBinaryOpVertex<>(this, those[0], (a, b) -> a.or(b));
        return new OrMultipleVertex(inputList(those));
    }

    @SafeVarargs
    public final BooleanVertex and(Vertex<BooleanTensor>... those) {
        if (those.length == 0) return this;
        if (those.length == 1) return new BooleanBinaryOpVertex<>(this, those[0], (a, b) -> a.and(b));
        return new AndMultipleVertex(inputList(those));
    }

    public static final BooleanVertex not(Vertex<BooleanTensor> vertex) {
        return new BooleanUnaryOpVertex<>(vertex, a -> a.not());
    }

    public BooleanVertex reshape(int... proposedShape) {
        return new BooleanUnaryOpVertex<>(this, a -> a.reshape(proposedShape));
    }

    public <T extends Tensor> BooleanVertex equalTo(Vertex<T> rhs) {
        return new BooleanBinaryOpVertex<>(this, rhs, (a, b) -> a.elementwiseEquals(b));
    }

    public <T extends Tensor> BooleanVertex notEqualTo(Vertex<T> rhs) {
        return new BooleanBinaryOpVertex<>(this, rhs, (a, b) -> a.elementwiseEquals(b).not());
    }

    private List<Vertex<BooleanTensor>> inputList(Vertex<BooleanTensor>[] those) {
        return ImmutableList.<Vertex<BooleanTensor>>builder()
            .addAll(Arrays.asList(those))
            .add(this)
            .build();
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
        this.observe(BooleanTensor.scalar(value));
    }

    public void observe(boolean[] values) {
        this.observe(BooleanTensor.create(values));
    }

    public boolean getValue(int... index) {
        return getValue().getValue(index);
    }

}
