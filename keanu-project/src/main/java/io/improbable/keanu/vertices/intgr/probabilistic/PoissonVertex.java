package io.improbable.keanu.vertices.intgr.probabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.Map;

import io.improbable.keanu.distributions.discrete.Poisson;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;

public class PoissonVertex extends IntegerVertex implements Probabilistic<IntegerTensor> {

    private final DoubleVertex mu;

    /**
     * One mu that must match a proposed tensor shape of Poisson.
     *
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param shape the desired shape of the vertex
     * @param mu    the mu of the Poisson with either the same shape as specified for this vertex or a scalar
     */
    // package private
    public PoissonVertex(int[] shape, DoubleVertex mu) {
        super(new ProbabilisticValueUpdater<>(), Observable.observableTypeFor(PoissonVertex.class));

        checkTensorsMatchNonScalarShapeOrAreScalar(shape, mu.getShape());

        this.mu = mu;
        setParents(mu);
        setValue(IntegerTensor.placeHolder(shape));
    }

    public Vertex<DoubleTensor> getMu() {
        return mu;
    }

    @Override
    public double logProb(IntegerTensor value) {
        return Poisson.withParameters(mu.getValue()).logProb(value).sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(IntegerTensor value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return Poisson.withParameters(mu.getValue()).sample(getShape(), random);
    }
}
