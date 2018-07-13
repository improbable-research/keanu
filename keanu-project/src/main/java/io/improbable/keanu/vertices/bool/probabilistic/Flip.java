package io.improbable.keanu.vertices.bool.probabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.Map;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;

public class Flip extends BoolVertex implements Probabilistic<BooleanTensor> {

    private final Vertex<DoubleTensor> probTrue;

    /**
     * One probTrue that must match a proposed tensor shape of Poisson.
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param shape    the desired shape of the vertex
     * @param probTrue the probability the flip returns true
     */
    public Flip(int[] shape, Vertex<DoubleTensor> probTrue) {
        super(new ProbabilisticValueUpdater<>(), Observable.observableTypeFor(Flip.class));

        checkTensorsMatchNonScalarShapeOrAreScalar(shape, probTrue.getShape());
        this.probTrue = probTrue;
        setParents(probTrue);
        setValue(BooleanTensor.placeHolder(shape));
    }

    /**
     * One to one constructor for mapping some shape of probTrue to
     * a matching shaped Flip.
     *
     * @param probTrue probTrue with same shape as desired Poisson tensor or scalar
     */
    public Flip(Vertex<DoubleTensor> probTrue) {
        this(probTrue.getShape(), probTrue);
    }

    public Flip(double probTrue) {
        this(Tensor.SCALAR_SHAPE, new ConstantDoubleVertex(probTrue));
    }

    public Flip(int[] shape, double probTrue) {
        this(shape, new ConstantDoubleVertex(probTrue));
    }

    public Vertex<DoubleTensor> getProbTrue() {
        return probTrue;
    }

    @Override
    public double logProb(BooleanTensor value) {

        DoubleTensor probTrueClamped = probTrue.getValue()
            .clamp(DoubleTensor.ZERO_SCALAR, DoubleTensor.ONE_SCALAR);

        DoubleTensor probability = value.setDoubleIf(
            probTrueClamped,
            probTrueClamped.unaryMinus().plusInPlace(1.0)
        );

        return probability.logInPlace().sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(BooleanTensor value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {

        DoubleTensor uniforms = random.nextDouble(this.getShape());

        return uniforms.lessThan(probTrue.getValue());
    }
}
