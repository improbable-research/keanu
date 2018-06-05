package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorStudentT;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.intgrtensor.IntegerVertex;
import io.improbable.keanu.vertices.intgrtensor.nonprobabilistic.ConstantIntegerVertex;

import java.util.HashMap;
import java.util.Map;

public class TensorStudentTVertex extends TensorProbabilisticDouble {

    private final IntegerVertex v;

    /**
     * @param shape expected tensor shape
     * @param v     Degrees of Freedom
     */
    public TensorStudentTVertex(int[] shape, IntegerVertex v) {
        this.v = v;
        setParents(v);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public TensorStudentTVertex(int[] shape, int v) {
        this(shape, new ConstantIntegerVertex(v));
    }

    public TensorStudentTVertex(IntegerVertex v) {
        this(v.getShape(), v);
    }

    public TensorStudentTVertex(int v) {
        this(Tensor.SCALAR_SHAPE, new ConstantIntegerVertex(v));
    }

    /**
     * @return degrees of freedom (v)
     */
    public IntegerVertex getV() {
        return v;
    }

    /**
     * @param t random variable
     * @return Log of the Probability Density of t
     */
    @Override
    public double logPdf(DoubleTensor t) {
        return TensorStudentT.logPdf(v.getValue(), t).sum();
    }

    /**
     * @param t random variable
     * @return Differential of the Log of the Probability Density of t
     */
    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor t) {
        TensorStudentT.Diff diff = TensorStudentT.dLogPdf(v.getValue(), t);
        Map<Long, DoubleTensor> m = new HashMap<>();
        m.put(getId(), diff.dPdt);
        return m;
    }

    /**
     * @return sample of Student T distribution
     */
    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return TensorStudentT.sample(getShape(), v.getValue(), random);
    }
}
