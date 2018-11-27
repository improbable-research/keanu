package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.discrete.Poisson;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
import io.improbable.keanu.vertices.SaveParentVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.CastDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class PoissonVertex extends IntegerVertex implements ProbabilisticInteger, SamplableWithManyScalars<IntegerTensor> {

    private final DoubleVertex mu;
    private static final String MU_NAME = "mu";

    /**
     * One mu that must match a proposed tensor shape of Poisson.
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param shape the desired shape of the vertex
     * @param mu    the mu of the Poisson with either the same shape as specified for this vertex or a scalar
     */
    public PoissonVertex(long[] shape, DoubleVertex mu) {
        super(shape);
        checkTensorsMatchNonScalarShapeOrAreScalar(shape, mu.getShape());

        this.mu = mu;
        setParents(mu);
    }

    public PoissonVertex(long[] shape, double mu) {
        this(shape, new ConstantDoubleVertex(mu));
    }

    /**
     * One to one constructor for mapping some shape of mu to
     * a matching shaped Poisson.
     *
     * @param mu mu with same shape as desired Poisson tensor or scalar
     */
    @ExportVertexToPythonBindings
    public PoissonVertex(@LoadParentVertex(MU_NAME) DoubleVertex mu) {
        this(mu.getShape(), mu);
    }

    public PoissonVertex(Vertex<? extends NumberTensor> mu) {
        this(mu.getShape(), new CastDoubleVertex(mu));
    }

    public PoissonVertex(double mu) {
        this(Tensor.SCALAR_SHAPE, new ConstantDoubleVertex(mu));
    }

    @SaveParentVertex(MU_NAME)
    public DoubleVertex getMu() {
        return mu;
    }

    @Override
    public double logProb(IntegerTensor value) {
        return Poisson.withParameters(mu.getValue()).logProb(value).sum();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(IntegerTensor value, Set<? extends Vertex> withRespectTo) {
        return Collections.emptyMap();
    }

    @Override
    public IntegerTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return Poisson.withParameters(mu.getValue()).sample(shape, random);
    }
}
