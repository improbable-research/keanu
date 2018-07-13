package io.improbable.keanu.vertices.bool;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.DiscreteVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.AndBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.OrBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.AndMultipleVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.OrMultipleVertex;
import io.improbable.keanu.vertices.update.ValueUpdater;

public abstract class BoolVertex extends DiscreteVertex<BooleanTensor> {

    public BoolVertex(ValueUpdater<BooleanTensor> valueUpdater) {
        super(valueUpdater);
    }

    @SafeVarargs
    public final io.improbable.keanu.vertices.bool.BoolVertex or(Vertex<BooleanTensor>... those) {
        if (those.length == 0) return this;
        if (those.length == 1) return new OrBinaryVertex(this, those[0]);
        return new OrMultipleVertex(inputList(those));
    }

    @SafeVarargs
    public final io.improbable.keanu.vertices.bool.BoolVertex and(Vertex<BooleanTensor>... those) {
        if (those.length == 0) return this;
        if (those.length == 1) return new AndBinaryVertex(this, those[0]);
        return new AndMultipleVertex(inputList(those));
    }

    private List<Vertex<BooleanTensor>> inputList(Vertex<BooleanTensor>[] those) {
        List<Vertex<BooleanTensor>> inputs = new LinkedList<>();
        inputs.addAll(Arrays.asList(those));
        inputs.add(this);
        return inputs;
    }

    public void setValue(boolean value) {
        super.setValue(BooleanTensor.create(value, getShape()));
    }

    public void setValue(boolean[] values) {
        super.setValue(BooleanTensor.create(values, getShape()));
    }

    public void setAndCascade(boolean value) {
        super.setAndCascade(BooleanTensor.create(value, getShape()));
    }

    public void setAndCascade(boolean[] values) {
        super.setAndCascade(BooleanTensor.create(values, getShape()));
    }

    public void observe(boolean value) {
        super.observe(BooleanTensor.create(value, getShape()));
    }

    public void observe(boolean[] values) {
        super.observe(BooleanTensor.create(values, getShape()));
    }

    public boolean getValue(int... index) {
        return getValue().getValue(index);
    }

}
