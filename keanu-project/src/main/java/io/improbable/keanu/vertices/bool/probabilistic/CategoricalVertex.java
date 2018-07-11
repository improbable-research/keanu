package io.improbable.keanu.vertices.bool.probabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class CategoricalVertex extends ProbabilisticBool {

    private final Vertex<DoubleTensor> probTrue;

    /**
     * One probTrue that must match a proposed tensor shape of Poisson.
     *
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param shape     the desired shape of the vertex
     * @param probTrue  the probability the flip returns true
     */
    public CategoricalVertex(int[] shape, Vertex<DoubleTensor> probTrue) {
        checkTensorsMatchNonScalarShapeOrAreScalar(shape, probTrue.getShape());
        this.probTrue = probTrue;
        setParents(probTrue);
        setValue(BooleanTensor.placeHolder(shape));
    }

    /**
     * One to one constructor for mapping some shape of probTrue to
     * a matching shaped CategoricalVertex.
     *
     * @param probTrue probTrue with same shape as desired Poisson tensor or scalar
     */
    public CategoricalVertex(Vertex<DoubleTensor> probTrue) {
        this(probTrue.getShape(), probTrue);
    }

    public CategoricalVertex(double probTrue) {
        this(Tensor.SCALAR_SHAPE, new ConstantDoubleVertex(probTrue));
    }

    public CategoricalVertex(int[] shape, double probTrue) {
        this(shape, new ConstantDoubleVertex(probTrue));
    }

    public Vertex<DoubleTensor> getProbTrue() {
        return probTrue;
    }

    @Override
    public double logPmf(BooleanTensor value) {

        DoubleTensor probability = value.setDoubleIf(
            probTrue.getValue(),
            probTrue.getValue().unaryMinus().plusInPlace(1.0)
        );

        return Math.log(probability.sum());
    }

    @Override
    public Map<Long, DoubleTensor> dLogPmf(BooleanTensor value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {

        DoubleTensor uniforms = random.nextDouble(this.getShape());

        return uniforms.lessThan(probTrue.getValue());
    }
}
