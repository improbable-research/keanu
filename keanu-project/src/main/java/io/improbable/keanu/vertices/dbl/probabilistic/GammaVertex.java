package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.Diffs.K;
import static io.improbable.keanu.distributions.dual.Diffs.THETA;
import static io.improbable.keanu.distributions.dual.Diffs.X;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import io.improbable.keanu.distributions.continuous.Gamma;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

public class GammaVertex extends DoubleVertex implements ProbabilisticDouble {

    private final DoubleVertex theta;
    private final DoubleVertex k;

    /**
     * Theta or k or both driving an arbitrarily shaped tensor of Gamma
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the vertex
     * @param theta       the theta (scale) of the Gamma with either the same shape as specified for this vertex
     * @param k           the k (shape) of the Gamma with either the same shape as specified for this vertex
     */
    public GammaVertex(int[] tensorShape, DoubleVertex theta, DoubleVertex k) {

        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, theta.getShape(), k.getShape());

        this.theta = theta;
        this.k = k;
        setParents(theta, k);
        setValue(DoubleTensor.placeHolder(tensorShape));
    }

    /**
     * One to one constructor for mapping some shape of theta and k to matching shaped gamma.
     *
     * @param theta the theta (scale) of the Gamma with either the same shape as specified for this vertex
     * @param k     the k (shape) of the Gamma with either the same shape as specified for this vertex
     */
    public GammaVertex(DoubleVertex theta, DoubleVertex k) {
        this(checkHasSingleNonScalarShapeOrAllScalar(theta.getShape(), k.getShape()), theta, k);
    }

    public GammaVertex(DoubleVertex theta, double k) {
        this(theta, new ConstantDoubleVertex(k));
    }

    public GammaVertex(double theta, DoubleVertex k) {
        this(new ConstantDoubleVertex(theta), k);
    }

    public GammaVertex(double theta, double k) {
        this(new ConstantDoubleVertex(theta), new ConstantDoubleVertex(k));
    }

    @Override
    public double logProb(DoubleTensor value) {
        DoubleTensor thetaValues = theta.getValue();
        DoubleTensor kValues = k.getValue();

        DoubleTensor logPdfs = Gamma.withParameters(thetaValues, kValues).logProb(value);
        return logPdfs.sum();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {
        Diffs dlnP = Gamma.withParameters(theta.getValue(), k.getValue()).dLogProb(value);

        Map<Vertex, DoubleTensor> dLogProbWrtParameters = new HashMap<>();

        if (withRespectTo.contains(theta)) {
            dLogProbWrtParameters.put(theta, dlnP.get(THETA).getValue());
        }

        if (withRespectTo.contains(k)) {
            dLogProbWrtParameters.put(k, dlnP.get(K).getValue());
        }

        if (withRespectTo.contains(this)) {
            dLogProbWrtParameters.put(this, dlnP.get(X).getValue());
        }

        return dLogProbWrtParameters;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return Gamma.withParameters(theta.getValue(), k.getValue()).sample(getShape(), random);
    }

}
