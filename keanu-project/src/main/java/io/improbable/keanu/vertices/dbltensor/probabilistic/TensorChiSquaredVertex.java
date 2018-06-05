package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorChiSquared;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.intgrtensor.IntegerVertex;
import io.improbable.keanu.vertices.intgrtensor.nonprobabilistic.ConstantIntegerVertex;

import java.util.Map;

public class TensorChiSquaredVertex extends TensorProbabilisticDouble {

    private IntegerVertex k;

    public TensorChiSquaredVertex(int[] shape, IntegerVertex k) {
        this.k = k;
        setParents(k);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public TensorChiSquaredVertex(int[] shape, int k) {
        this(shape, new ConstantIntegerVertex(k));
    }

    public TensorChiSquaredVertex(IntegerTensor k) {
        this(k.getShape(), new ConstantIntegerVertex(k));
    }

    public TensorChiSquaredVertex(int k) {
        this(Tensor.SCALAR_SHAPE, new ConstantIntegerVertex(k));
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return TensorChiSquared.sample(getShape(), k.getValue(), random);
    }

    @Override
    public double logPdf(DoubleTensor value) {
        return TensorChiSquared.logPdf(k.getValue(), value).sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        throw new UnsupportedOperationException();
    }

}
