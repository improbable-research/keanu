package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorStudentT;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class StudentTVertex extends ProbabilisticDouble {

    private final IntegerVertex v;

    /**
     * One v driving an arbitrarily shaped tensor of StudentT
     *
     * @param shape the desired shape of the vertex
     * @param v     the number of degrees of freedom
     */
    public StudentTVertex(int[] shape, IntegerVertex v) {
        checkTensorsMatchNonScalarShapeOrAreScalar(shape, v.getShape());
        this.v = v;
        setParents(v);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public StudentTVertex(int[] shape, int v) {
        this(shape, new ConstantIntegerVertex(v));
    }

    public StudentTVertex(IntegerVertex v) {
        this(v.getShape(), v);
    }

    public StudentTVertex(int v) {
        this(Tensor.SCALAR_SHAPE, new ConstantIntegerVertex(v));
    }

    public IntegerVertex getV() {
        return v;
    }

    @Override
    public double logPdf(DoubleTensor t) {
        return TensorStudentT.logPdf(v.getValue(), t).sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor t) {
        TensorStudentT.Diff diff = TensorStudentT.dLogPdf(v.getValue(), t);
        Map<Long, DoubleTensor> m = new HashMap<>();
        m.put(getId(), diff.dPdt);
        return m;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return TensorStudentT.sample(getShape(), v.getValue(), random);
    }
}
