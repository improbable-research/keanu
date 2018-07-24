package io.improbable.keanu.vertices.bool.probabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class Flip extends ProbabilisticBool {

    private final DoubleVertex probTrue;

    /**
     * One probTrue that must match a proposed tensor shape of Poisson.
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param shape    the desired shape of the vertex
     * @param probTrue the probability the flip returns true
     */
    public Flip(int[] shape, DoubleVertex probTrue) {
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
    public Flip(DoubleVertex probTrue) {
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
    public double logPmf(BooleanTensor value) {

        DoubleTensor probability = value.setDoubleIf(
            probTrue.getValue(),
            probTrue.getValue().unaryMinus().plusInPlace(1.0)
        ).clampInPlace(DoubleTensor.ZERO_SCALAR, DoubleTensor.ONE_SCALAR);

        return probability.logInPlace().sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPmf(BooleanTensor value) {

        DualNumber probTrueDual = probTrue.getDualNumber();

        DoubleTensor dLogPdp = value.setDoubleIf(
            DoubleTensor.ONE_SCALAR,
            DoubleTensor.MINUS_ONE_SCALAR
        );

        PartialDerivatives partials = probTrueDual
            .getPartialDerivatives()
            .multiplyBy(dLogPdp)
            .sum(true, TensorShape.dimensionRange(0, value.getRank()));

        return partials.asMap();
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {

        DoubleTensor uniforms = random.nextDouble(this.getShape());

        return uniforms.lessThan(probTrue.getValue());
    }
}
