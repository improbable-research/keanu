package io.improbable.keanu.vertices.bool.probabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import io.improbable.keanu.distributions.discrete.Bernoulli;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import java.util.Collections;
import java.util.Map;
import java.util.Set;

public class BernoulliVertex extends BoolVertex implements ProbabilisticBoolean {

    private final Vertex<DoubleTensor> probTrue;

    /**
     * One probTrue that must match a proposed tensor shape of Bernoulli.
     *
     * <p>If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param shape the desired shape of the vertex
     * @param probTrue the probability the bernoulli returns true
     */
    public BernoulliVertex(int[] shape, Vertex<DoubleTensor> probTrue) {
        checkTensorsMatchNonScalarShapeOrAreScalar(shape, probTrue.getShape());
        this.probTrue = probTrue;
        setParents(probTrue);
        setValue(BooleanTensor.placeHolder(shape));
    }

    /**
     * One to one constructor for mapping some shape of probTrue to a matching shaped Bernoulli.
     *
     * @param probTrue probTrue with same shape as desired Bernoulli tensor or scalar
     */
    public BernoulliVertex(Vertex<DoubleTensor> probTrue) {
        this(probTrue.getShape(), probTrue);
    }

    public BernoulliVertex(double probTrue) {
        this(Tensor.SCALAR_SHAPE, new ConstantDoubleVertex(probTrue));
    }

    public BernoulliVertex(int[] shape, double probTrue) {
        this(shape, new ConstantDoubleVertex(probTrue));
    }

    public Vertex<DoubleTensor> getProbTrue() {
        return probTrue;
    }

    @Override
    public double logProb(BooleanTensor value) {
        return Bernoulli.withParameters(probTrue.getValue()).logProb(value).sum();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(
            BooleanTensor value, Set<? extends Vertex> withRespectTo) {

        if (!(probTrue instanceof Differentiable)) {
            throw new UnsupportedOperationException(
                    "The probability of the Bernoulli being true must be differentiable");
        }

        if (withRespectTo.contains(probTrue)) {
            DoubleTensor dLogPdp = Bernoulli.withParameters(probTrue.getValue()).dLogProb(value);
            return Collections.singletonMap(probTrue, dLogPdp);
        }

        return Collections.emptyMap();
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return Bernoulli.withParameters(probTrue.getValue()).sample(this.getShape(), random);
    }
}
