package io.improbable.keanu.vertices.bool;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.AndBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.OrBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.AndMultipleVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.OrMultipleVertex;
import io.improbable.keanu.vertices.update.ValueUpdater;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BoolTakeVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BoolSliceVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.NotVertex;

public abstract class BoolVertex extends Vertex<BooleanTensor> {

    public BoolVertex(ValueUpdater<BooleanTensor> valueUpdater, Observable<BooleanTensor> observation) {
        super(valueUpdater, observation);
    }

    @SafeVarargs
    public final BoolVertex or(Vertex<BooleanTensor>... those) {
        if (those.length == 0) return this;
        if (those.length == 1) return new OrBinaryVertex(this, those[0]);
        return new OrMultipleVertex(inputList(those));
    }

    @SafeVarargs
    public final BoolVertex and(Vertex<BooleanTensor>... those) {
        if (those.length == 0) return this;
        if (those.length == 1) return new AndBinaryVertex(this, those[0]);
        return new AndMultipleVertex(inputList(those));
    }

    public static final BoolVertex not(Vertex<BooleanTensor> vertex) {
        return new NotVertex(vertex);
    }

    private List<Vertex<BooleanTensor>> inputList(Vertex<BooleanTensor>[] those) {
        List<Vertex<BooleanTensor>> inputs = new LinkedList<>();
        inputs.addAll(Arrays.asList(those));
        inputs.add(this);
        return inputs;
    }

    public BoolVertex slice(int dimension, int index) {
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

    public boolean getValue(int... index) {
        return getValue().getValue(index);
    }

    public BoolVertex take(int... index) {
        return new BoolTakeVertex(this, index);
    }


}
