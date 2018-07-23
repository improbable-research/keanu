package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.ParameterName.T;

import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.distributions.continuous.StudentT;
import io.improbable.keanu.distributions.dual.ParameterMap;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;

public class StudentTVertex extends DistributionBackedDoubleVertex<IntegerVertex, IntegerTensor> {

    private final IntegerVertex v;

    /**
     * One v that must match a proposed tensor shape of StudentT
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape expected tensor shape
     * @param v           Degrees of Freedom
     */
    // package private
    StudentTVertex(int[] tensorShape, IntegerVertex v) {
        super(tensorShape, StudentT::withParameters, v);
        this.v = v;
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor t) {
        ParameterMap<DoubleTensor> diff = StudentT.withParameters(v.getValue()).dLogProb(t);
        Map<Long, DoubleTensor> m = new HashMap<>();
        m.put(getId(), diff.get(T).getValue());
        return m;
    }
}
