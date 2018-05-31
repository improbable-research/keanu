package io.improbable.keanu.vertices.intgrtensor.probabilistic;

import io.improbable.keanu.distributions.tensors.discrete.Poisson;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.CastDoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantDoubleTensorVertex;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class PoissonVertex extends ProbabilisticInteger {

    private final DoubleTensorVertex mu;

    public PoissonVertex(int[] shape, DoubleTensorVertex mu) {

        checkTensorsMatchNonScalarShapeOrAreScalar(shape, mu.getShape());

        this.mu = mu;
        setParents(mu);
        setValue(IntegerTensor.placeHolder(shape));
    }

    public PoissonVertex(int[] shape, double mu) {
        this(shape, new ConstantDoubleTensorVertex(mu));
    }

    public PoissonVertex(DoubleTensorVertex mu) {
        this(mu.getShape(), mu);
    }

    public PoissonVertex(Vertex<? extends NumberTensor> mu) {
        this(mu.getShape(), new CastDoubleTensorVertex(mu));
    }

    public PoissonVertex(double mu) {
        this(Tensor.SCALAR_SHAPE, new ConstantDoubleTensorVertex(mu));
    }

    public Vertex<DoubleTensor> getMu() {
        return mu;
    }

    @Override
    public double logPmf(IntegerTensor value) {
        return Poisson.logPmf(mu.getValue(), value).sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPmf(IntegerTensor value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return Poisson.sample(getShape(), mu.getValue(), random);
    }
}
