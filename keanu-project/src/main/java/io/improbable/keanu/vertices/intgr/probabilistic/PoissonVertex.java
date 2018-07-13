package io.improbable.keanu.vertices.intgr.probabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.Map;

import io.improbable.keanu.distributions.discrete.Poisson;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Observation;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.CastDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
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
    public PoissonVertex(int[] shape, DoubleVertex mu) {
        super(new ProbabilisticValueUpdater<>(), new Observation<>());

        checkTensorsMatchNonScalarShapeOrAreScalar(shape, mu.getShape());

        this.mu = mu;
        setParents(mu);
        setValue(IntegerTensor.placeHolder(shape));
    }

    public PoissonVertex(int[] shape, double mu) {
        this(shape, new ConstantDoubleVertex(mu));
    }

    /**
     * One to one constructor for mapping some shape of mu to
     * a matching shaped Poisson.
     *
     * @param mu    mu with same shape as desired Poisson tensor or scalar
     */
    public PoissonVertex(DoubleVertex mu) {
        this(mu.getShape(), mu);
    }

    public PoissonVertex(Vertex<? extends NumberTensor> mu) {
        this(mu.getShape(), new CastDoubleVertex(mu));
    }

    public PoissonVertex(double mu) {
        this(Tensor.SCALAR_SHAPE, new ConstantDoubleVertex(mu));
    }

    public Vertex<DoubleTensor> getMu() {
        return mu;
    }

    @Override
    public double logProb(IntegerTensor value) {
        return Poisson.withParamters(mu.getValue()).logProb(value).sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(IntegerTensor value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return Poisson.withParamters(mu.getValue()).sample(getShape(), random);
    }
}
