package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;
import io.improbable.keanu.vertices.update.ValueUpdater;

import java.util.Map;
import java.util.function.Function;
import java.util.function.Supplier;

public abstract class NonProbabilisticBool extends BoolVertex {

    private Function<Vertex<BooleanTensor>, BooleanTensor> updateFunction;

    public NonProbabilisticBool(Function<Vertex<BooleanTensor>, BooleanTensor> updateFunction) {
        super(new NonProbabilisticValueUpdater<>(updateFunction));
        this.updateFunction = updateFunction;
    }

    @Override
    public double logPmf(BooleanTensor value) {
        return updateFunction.apply(this).elementwiseEquals(value).allTrue() ? 0.0 : Double.NEGATIVE_INFINITY;
    }

    @Override
    public Map<Long, DoubleTensor> dLogPmf(BooleanTensor value) {
        throw new UnsupportedOperationException();
    }

}
