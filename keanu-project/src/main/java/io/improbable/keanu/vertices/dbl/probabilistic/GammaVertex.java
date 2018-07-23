package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.ParameterName.A;
import static io.improbable.keanu.distributions.dual.ParameterName.K;
import static io.improbable.keanu.distributions.dual.ParameterName.THETA;
import static io.improbable.keanu.distributions.dual.ParameterName.X;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.Map;

import io.improbable.keanu.distributions.continuous.Gamma;
import io.improbable.keanu.distributions.dual.ParameterMap;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;

public class GammaVertex extends DoubleVertex implements Probabilistic<DoubleTensor> {

    private final DoubleVertex location;
    private final DoubleVertex theta;
    private final DoubleVertex k;

    /**
     * One location, theta or k or all three driving an arbitrarily shaped tensor of Gamma
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the vertex
     * @param location    the location of the Gamma with either the same shape as specified for this vertex or location scalar
     * @param theta       the theta (scale) of the Gamma with either the same shape as specified for this vertex or location scalar
     * @param k           the k (shape) of the Gamma with either the same shape as specified for this vertex or location scalar
     */
    public GammaVertex(int[] tensorShape, DoubleVertex location, DoubleVertex theta, DoubleVertex k) {
        super(new ProbabilisticValueUpdater<>(), Observable.observableTypeFor(GammaVertex.class));

        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, location.getShape(), theta.getShape(), k.getShape());

        this.location = location;
        this.theta = theta;
        this.k = k;
        setParents(location, theta, k);
        setValue(DoubleTensor.placeHolder(tensorShape));
    }

    /**
     * One to one constructor for mapping some shape of location, theta and k to
     * location matching shaped gamma.
     *
     * @param location the location of the Gamma with either the same shape as specified for this vertex or location scalar
     * @param theta    the theta (scale) of the Gamma with either the same shape as specified for this vertex or location scalar
     * @param k        the k (shape) of the Gamma with either the same shape as specified for this vertex or location scalar
     */
    public GammaVertex(DoubleVertex location, DoubleVertex theta, DoubleVertex k) {
        this(checkHasSingleNonScalarShapeOrAllScalar(location.getShape(), theta.getShape(), k.getShape()), location, theta, k);
    }

    public GammaVertex(DoubleVertex location, DoubleVertex theta, double k) {
        this(location, theta, new ConstantDoubleVertex(k));
    }

    public GammaVertex(DoubleVertex location, double theta, DoubleVertex k) {
        this(location, new ConstantDoubleVertex(theta), k);
    }

    public GammaVertex(DoubleVertex location, double theta, double k) {
        this(location, new ConstantDoubleVertex(theta), new ConstantDoubleVertex(k));
    }

    public GammaVertex(double location, DoubleVertex theta, DoubleVertex k) {
        this(new ConstantDoubleVertex(location), theta, k);
    }

    public GammaVertex(double location, DoubleVertex theta, double k) {
        this(new ConstantDoubleVertex(location), theta, new ConstantDoubleVertex(k));
    }

    public GammaVertex(double location, double theta, DoubleVertex k) {
        this(new ConstantDoubleVertex(location), new ConstantDoubleVertex(theta), k);
    }

    public GammaVertex(double location, double theta, double k) {
        this(new ConstantDoubleVertex(location), new ConstantDoubleVertex(theta), new ConstantDoubleVertex(k));
    }

    @Override
    public double logProb(DoubleTensor value) {
        DoubleTensor locationValues = location.getValue();
        DoubleTensor thetaValues = theta.getValue();
        DoubleTensor kValues = k.getValue();

        DoubleTensor logPdfs = Gamma.withParameters(locationValues, thetaValues, kValues).logProb(value);
        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {
        ParameterMap<DoubleTensor> dlnP = Gamma.withParameters(location.getValue(), theta.getValue(), k.getValue()).dLogProb(value);

        return convertDualNumbersToDiff(dlnP.get(A).getValue(), dlnP.get(THETA).getValue(), dlnP.get(K).getValue(), dlnP.get(X).getValue());

    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dPdlocation,
                                                             DoubleTensor dPdtheta,
                                                             DoubleTensor dPdk,
                                                             DoubleTensor dPdx) {

        Differentiator differentiator = new Differentiator();
        PartialDerivatives dPdInputsFromA = differentiator.calculateDual((Differentiable) location).getPartialDerivatives().multiplyBy(dPdlocation);
        PartialDerivatives dPdInputsFromTheta = differentiator.calculateDual((Differentiable) theta).getPartialDerivatives().multiplyBy(dPdtheta);
        PartialDerivatives dPdInputsFromK = differentiator.calculateDual((Differentiable) k).getPartialDerivatives().multiplyBy(dPdk);
        PartialDerivatives dPdInputs = dPdInputsFromA.add(dPdInputsFromTheta).add(dPdInputsFromK);

        if (!this.isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return dPdInputs.asMap();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return Gamma.withParameters(location.getValue(), theta.getValue(), k.getValue()).sample(getShape(), random);
    }

}
