package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.Diffs.MU;
import static io.improbable.keanu.distributions.dual.Diffs.SIGMA;
import static io.improbable.keanu.distributions.dual.Diffs.X;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import io.improbable.keanu.distributions.continuous.LogNormal;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class LogNormalVertex extends DoubleVertex implements ProbabilisticDouble {

    private final DoubleVertex mu;
    private final DoubleVertex sigma;

    /**
     * One mu or s or both driving an arbitrarily shaped tensor of LogNormal
     * https://en.wikipedia.org/wiki/Log-normal_distribution
     *
     * @param tensorShape the desired shape of the vertex
     * @param mu the mu (location) of the LogNormal with either the same tensor shape as specified
     *     for this vertex or mu scalar
     * @param sigma the sigma of the Logistic with either the same shape as specified for this
     *     vertex or mu scalar
     */
    public LogNormalVertex(int[] tensorShape, DoubleVertex mu, DoubleVertex sigma) {

        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, mu.getShape(), sigma.getShape());

        this.mu = mu;
        this.sigma = sigma;
        setParents(mu, sigma);
        setValue(DoubleTensor.placeHolder(tensorShape));
    }

    public LogNormalVertex(int[] tensorShape, DoubleVertex mu, double sigma) {
        this(tensorShape, mu, ConstantVertex.of(sigma));
    }

    public LogNormalVertex(int[] tensorShape, double mu, DoubleVertex sigma) {
        this(tensorShape, ConstantVertex.of(mu), sigma);
    }

    public LogNormalVertex(int[] tensorShape, double mu, double sigma) {
        this(tensorShape, ConstantVertex.of(mu), ConstantVertex.of(sigma));
    }

    public LogNormalVertex(DoubleVertex mu, DoubleVertex sigma) {
        this(checkHasSingleNonScalarShapeOrAllScalar(mu.getShape(), sigma.getShape()), mu, sigma);
    }

    public LogNormalVertex(double mu, DoubleVertex sigma) {
        this(ConstantVertex.of(mu), sigma);
    }

    public LogNormalVertex(DoubleVertex mu, double sigma) {
        this(mu, ConstantVertex.of(sigma));
    }

    public LogNormalVertex(double mu, double sigma) {
        this(ConstantVertex.of(mu), ConstantVertex.of(sigma));
    }

    @Override
    public double logProb(DoubleTensor value) {
        DoubleTensor muValues = mu.getValue();
        DoubleTensor sigmaValues = sigma.getValue();

        DoubleTensor logPdfs = LogNormal.withParameters(muValues, sigmaValues).logProb(value);

        return logPdfs.sum();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(
            DoubleTensor value, Set<? extends Vertex> withRespectTo) {
        Diffs dlnP = LogNormal.withParameters(mu.getValue(), sigma.getValue()).dLogProb(value);

        Map<Vertex, DoubleTensor> dLogProbWrtParameters = new HashMap<>();

        if (withRespectTo.contains(mu)) {
            dLogProbWrtParameters.put(mu, dlnP.get(MU).getValue());
        }

        if (withRespectTo.contains(sigma)) {
            dLogProbWrtParameters.put(sigma, dlnP.get(SIGMA).getValue());
        }

        if (withRespectTo.contains(this)) {
            dLogProbWrtParameters.put(this, dlnP.get(X).getValue());
        }

        return dLogProbWrtParameters;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return LogNormal.withParameters(mu.getValue(), sigma.getValue()).sample(getShape(), random);
    }
}
