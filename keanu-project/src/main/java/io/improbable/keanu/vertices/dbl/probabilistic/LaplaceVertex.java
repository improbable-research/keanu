package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.Laplace;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
import io.improbable.keanu.vertices.SaveParentVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.distributions.hyperparam.Diffs.BETA;
import static io.improbable.keanu.distributions.hyperparam.Diffs.MU;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class LaplaceVertex extends DoubleVertex implements Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor> {

    private final DoubleVertex mu;
    private final DoubleVertex beta;
    private static final String MU_NAME = "mu";
    private static final String BETA_NAME = "beta";

    /**
     * One mu or beta or both that match a proposed tensor shape of Laplace
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the tensor within the vertex
     * @param mu          the mu of the Laplace with either the same shape as specified for this vertex or a scalar
     * @param beta        the beta of the Laplace with either the same shape as specified for this vertex or a scalar
     */
    public LaplaceVertex(long[] tensorShape, DoubleVertex mu, DoubleVertex beta) {
        super(tensorShape);
        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, mu.getShape(), beta.getShape());

        this.mu = mu;
        this.beta = beta;
        setParents(mu, beta);
    }

    public LaplaceVertex(long[] shape, DoubleVertex mu, double beta) {
        this(shape, mu, new ConstantDoubleVertex(beta));
    }

    public LaplaceVertex(long[] shape, double mu, DoubleVertex beta) {
        this(shape, new ConstantDoubleVertex(mu), beta);
    }

    public LaplaceVertex(long[] shape, double mu, double beta) {
        this(shape, new ConstantDoubleVertex(mu), new ConstantDoubleVertex(beta));
    }

    /**
     * One to one constructor for mapping some shape of mu and sigma to
     * a matching shaped Laplace.
     *
     * @param mu   the mu of the Laplace with either the same shape as specified for this vertex or a scalar
     * @param beta the beta of the Laplace with either the same shape as specified for this vertex or a scalar
     */
    @ExportVertexToPythonBindings
    public LaplaceVertex(@LoadParentVertex(MU_NAME) DoubleVertex mu,
                         @LoadParentVertex(BETA_NAME) DoubleVertex beta) {
        this(checkHasSingleNonScalarShapeOrAllScalar(mu.getShape(), beta.getShape()), mu, beta);
    }

    public LaplaceVertex(DoubleVertex mu, double beta) {
        this(mu, new ConstantDoubleVertex(beta));
    }

    public LaplaceVertex(double mu, DoubleVertex beta) {
        this(new ConstantDoubleVertex(mu), beta);
    }

    public LaplaceVertex(double mu, double beta) {
        this(new ConstantDoubleVertex(mu), new ConstantDoubleVertex(beta));
    }

    @SaveParentVertex(MU_NAME)
    public DoubleVertex getMu() {
        return mu;
    }

    @SaveParentVertex(BETA_NAME)
    public DoubleVertex getBeta() {
        return beta;
    }

    @Override
    public double logProb(DoubleTensor value) {

        DoubleTensor muValues = mu.getValue();
        DoubleTensor betaValues = beta.getValue();

        DoubleTensor logPdfs = Laplace.withParameters(muValues, betaValues).logProb(value);

        return logPdfs.sum();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {
        Diffs dlnP = Laplace.withParameters(mu.getValue(), beta.getValue()).dLogProb(value);

        Map<Vertex, DoubleTensor> dLogProbWrtParameters = new HashMap<>();

        if (withRespectTo.contains(mu)) {
            dLogProbWrtParameters.put(mu, dlnP.get(MU).getValue());
        }

        if (withRespectTo.contains(beta)) {
            dLogProbWrtParameters.put(beta, dlnP.get(BETA).getValue());
        }

        if (withRespectTo.contains(this)) {
            dLogProbWrtParameters.put(this, dlnP.get(X).getValue());
        }

        return dLogProbWrtParameters;
    }

    @Override
    public DoubleTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return Laplace.withParameters(mu.getValue(), beta.getValue()).sample(shape, random);
    }

}
