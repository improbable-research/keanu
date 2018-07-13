package io.improbable.keanu.vertices.intgr.nonprobabilistic;

import java.util.function.Function;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilisticObservationException;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public abstract class NonProbabilisticInteger extends IntegerVertex {

    private Function<Vertex<IntegerTensor>, IntegerTensor> updateFunction;

    public NonProbabilisticInteger(Function<Vertex<IntegerTensor>, IntegerTensor> updateFunction) {
        super(new NonProbabilisticValueUpdater<>(updateFunction));
        this.updateFunction = updateFunction;
    }

    /**
     * Observing non-probabilistic values of this type causes the probability
     * of the graph to flatten to 0 in all places that doesn't exactly match
     * the observation. This is so bad that it is actually prohibited by throwing
     * an exception. This is not the case for all types of non-probabilistic
     * observations.
     *
     * @param value the value to be observed
     */
    @Override
    public void observe(IntegerTensor value) {
        throw new NonProbabilisticObservationException();
    }

    @Override
    public boolean matchesObservation() {
        return updateFunction.apply(this).elementwiseEquals(getValue()).allTrue();
    }
}
