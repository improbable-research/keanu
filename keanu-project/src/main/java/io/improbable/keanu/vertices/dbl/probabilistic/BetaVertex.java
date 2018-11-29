package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.Beta;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.LogProbGraph;
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

import static io.improbable.keanu.distributions.hyperparam.Diffs.A;
import static io.improbable.keanu.distributions.hyperparam.Diffs.B;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;

public class BetaVertex extends DoubleVertex implements Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor> {

    private final DoubleVertex alpha;
    private final DoubleVertex beta;
    private static final String ALPHA_NAME = "alpha";
    private static final String BETA_NAME = "beta";

    private final Beta distribution;

    /**
     * One alpha or beta or both that match a proposed tensor shape of Beta.
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the tensor contained in the vertex
     * @param alpha       the alpha of the Beta with either the same tensorShape as specified for this vertex or a scalar
     * @param beta        the beta of the Beta with either the same tensorShape as specified for this vertex or a scalar
     */
    public BetaVertex(long[] tensorShape, DoubleVertex alpha, DoubleVertex beta) {
        super(tensorShape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(tensorShape, alpha.getShape(), beta.getShape());

        this.alpha = alpha;
        this.beta = beta;
        this.distribution = Beta.withParameters(this, alpha, beta, ConstantVertex.of(0.0), ConstantVertex.of(1.0));
        setParents(alpha, beta);
    }

    /**
     * One to one constructor for mapping some tensorShape of alpha and beta to
     * a matching tensorShaped Beta.
     *
     * @param alpha the alpha of the Beta with either the same tensorShape as specified for this vertex or a scalar
     * @param beta  the beta of the Beta with either the same tensorShape as specified for this vertex or a scalar
     */
    @ExportVertexToPythonBindings
    public BetaVertex(@LoadParentVertex(ALPHA_NAME) DoubleVertex alpha,
                      @LoadParentVertex(BETA_NAME) DoubleVertex beta) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(alpha.getShape(), beta.getShape()), alpha, beta);
    }

    public BetaVertex(DoubleVertex alpha, double beta) {
        this(alpha, new ConstantDoubleVertex(beta));
    }

    public BetaVertex(double alpha, DoubleVertex beta) {
        this(new ConstantDoubleVertex(alpha), beta);
    }

    public BetaVertex(double alpha, double beta) {
        this(new ConstantDoubleVertex(alpha), new ConstantDoubleVertex(beta));
    }

    public BetaVertex(long[] tensorShape, DoubleVertex alpha, double beta) {
        this(tensorShape, alpha, new ConstantDoubleVertex(beta));
    }

    public BetaVertex(long[] tensorShape, double alpha, DoubleVertex beta) {
        this(tensorShape, new ConstantDoubleVertex(alpha), beta);
    }

    public BetaVertex(long[] tensorShape, double alpha, double beta) {
        this(tensorShape, new ConstantDoubleVertex(alpha), new ConstantDoubleVertex(beta));
    }

    @SaveParentVertex(ALPHA_NAME)
    public DoubleVertex getAlpha() {
        return alpha;
    }

    @SaveParentVertex(BETA_NAME)
    public DoubleVertex getBeta() {
        return beta;
    }

    @Override
    public double logProb(DoubleTensor value) {
        return distribution.logProb(value).sum();
    }

    public LogProbGraph logProbGraph() {
        return distribution.logProbGraph();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {
        Diffs dlnP = distribution.dLogProb(value);

        Map<Vertex, DoubleTensor> dLogProbWrtParameters = new HashMap<>();

        if (withRespectTo.contains(alpha)) {
            dLogProbWrtParameters.put(alpha, dlnP.get(A).getValue());
        }

        if (withRespectTo.contains(beta)) {
            dLogProbWrtParameters.put(beta, dlnP.get(B).getValue());
        }

        if (withRespectTo.contains(this)) {
            dLogProbWrtParameters.put(this, dlnP.get(X).getValue());
        }

        return dLogProbWrtParameters;
    }

    @Override
    public DoubleTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return distribution.sample(shape, random);
    }

}
