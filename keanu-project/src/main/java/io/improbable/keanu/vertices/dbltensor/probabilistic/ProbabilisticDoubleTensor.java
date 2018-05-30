package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorDualNumber;

import java.util.Map;

public abstract class ProbabilisticDoubleTensor extends DoubleTensorVertex {

    @Override
    public DoubleTensor updateValue() {
        if (!hasValue()) {
            setValue(sampleUsingDefaultRandom());
        }
        return getValue();
    }

    @Override
    public boolean isProbabilistic() {
        return true;
    }

    @Override
    protected TensorDualNumber calculateDualNumber(Map<Vertex, TensorDualNumber> dualNumbers) {
        if (isObserved()) {
            return TensorDualNumber.createConstant(getValue());
        } else {
            return TensorDualNumber.createWithRespectToSelf(getId(), getValue());
        }
    }

}
