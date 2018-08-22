package io.improbable.keanu.distributions;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

public class IntegerSupport extends Support<IntegerTensor> {
    public IntegerSupport(IntegerTensor min, IntegerTensor max, int[] shape) {
        super(min, max, shape);
    }

    @Override
    public boolean isSubsetOf(Support<IntegerTensor> q) {
        return getMin().greaterThanOrEqual(q.getMin()).allTrue() && getMax().lessThanOrEqual(q.getMax()).allTrue();
    }
}
