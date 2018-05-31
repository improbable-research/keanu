package io.improbable.keanu.vertices.booltensor.probabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.booltensor.probabilistic.ProbabilisticBool;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantDoubleTensorVertex;

import java.util.Map;

public class Flip extends ProbabilisticBool {

    private final Vertex<DoubleTensor> probTrue;

    public Flip(int[] shape, Vertex<DoubleTensor> probTrue) {
        this.probTrue = probTrue;
        setParents(probTrue);
        setValue(BooleanTensor.placeHolder(shape));
    }

    public Flip(Vertex<DoubleTensor> probTrue) {
        this(Tensor.SCALAR_SHAPE, probTrue);
    }

    public Flip(double probTrue) {
        this(Tensor.SCALAR_SHAPE, new ConstantDoubleTensorVertex(probTrue));
    }

    public Flip(int[] shape, double probTrue) {
        this(shape, new ConstantDoubleTensorVertex(probTrue));
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

        DoubleTensor uniforms = random.nextDouble(probTrue.getValue().getShape());

        return uniforms.lessThan(probTrue.getValue());
    }

}
