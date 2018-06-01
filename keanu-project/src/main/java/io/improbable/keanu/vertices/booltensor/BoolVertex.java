package io.improbable.keanu.vertices.booltensor;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.DiscreteTensorVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.booltensor.nonprobabilistic.ConstantBoolVertex;
import io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.binary.AndBinaryVertex;
import io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.binary.OrBinaryVertex;
import io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.multiple.AndMultipleVertex;
import io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.multiple.OrMultipleVertex;
import io.improbable.keanu.vertices.generictensor.nonprobabilistic.ConstantVertex;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public abstract class BoolVertex extends DiscreteTensorVertex<BooleanTensor> {

    @SafeVarargs
    public final io.improbable.keanu.vertices.booltensor.BoolVertex or(Vertex<BooleanTensor>... those) {
        if (those.length == 0) return this;
        if (those.length == 1) return new OrBinaryVertex(this, those[0]);
        return new OrMultipleVertex(inputList(those));
    }

    @SafeVarargs
    public final io.improbable.keanu.vertices.booltensor.BoolVertex and(Vertex<BooleanTensor>... those) {
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
        super.setValue(BooleanTensor.scalar(value));
    }

    public void setAndCascade(boolean value) {
        super.setAndCascade(BooleanTensor.scalar(value));
    }

    public void setAndCascade(boolean value, Map<Long, Long> explored) {
        super.setAndCascade(BooleanTensor.scalar(value), explored);
    }

    public void observe(boolean value) {
        super.observe(BooleanTensor.scalar(value));
    }

    public double logPmf(boolean value) {
        if (this.getValue().isScalar()) {
            return this.logPmf(BooleanTensor.scalar(value));
        } else {
            throw new IllegalArgumentException("Vertex is not scalar");
        }
    }

    public Map<Long, DoubleTensor> dLogPmf(boolean value) {
        if (this.getValue().isScalar()) {
            return this.dLogPmf(BooleanTensor.scalar(value));
        } else {
            throw new IllegalArgumentException("Vertex is not scalar");
        }
    }

}
