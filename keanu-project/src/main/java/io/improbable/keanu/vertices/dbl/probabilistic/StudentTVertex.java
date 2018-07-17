package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.Diffs.T;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.distributions.continuous.StudentT;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;

public class StudentTVertex extends ProbabilisticDouble {

    private final IntegerVertex v;

    /**
     * One v that must match a proposed tensor shape of StudentT
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape expected tensor shape
     * @param v           Degrees of Freedom
     */
    public StudentTVertex(int[] tensorShape, IntegerVertex v) {
        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, v.getShape());
        this.v = v;
        setParents(v);
        setValue(DoubleTensor.placeHolder(tensorShape));
    }

    public StudentTVertex(int[] tensorShape, int v) {
        this(tensorShape, new ConstantIntegerVertex(v));
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
        return StudentT.withParameters(v.getValue()).logProb(t).sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor t) {
        Diffs diff = StudentT.withParameters(v.getValue()).dLogProb(t);
        Map<Long, DoubleTensor> m = new HashMap<>();
        m.put(getId(), diff.get(T).getValue());
        return m;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return StudentT.withParameters(v.getValue()).sample(getShape(), random);
    }
}
