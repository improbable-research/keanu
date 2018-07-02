package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Gamma;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class GammaVertex extends ProbabilisticDouble {

    private final DoubleVertex location;
    private final DoubleVertex theta;
    private final DoubleVertex k;

    /**
     * One location, theta or k or all three driving an arbitrarily shaped tensor of Gamma
     *
     * @param tensorShape the desired shape of the vertex
     * @param location    the location of the Gamma with either the same shape as specified for this vertex or location scalar
     * @param theta       the theta (scale) of the Gamma with either the same shape as specified for this vertex or location scalar
     * @param k           the k (shape) of the Gamma with either the same shape as specified for this vertex or location scalar
     */
    public GammaVertex(int[] tensorShape, DoubleVertex location, DoubleVertex theta, DoubleVertex k) {
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
    public double logPdf(DoubleTensor value) {
        DoubleTensor locationValues = location.getValue();
        DoubleTensor thetaValues = theta.getValue();
        DoubleTensor kValues = k.getValue();

        DoubleTensor logPdfs = Gamma.logPdf(locationValues, thetaValues, kValues, value);
        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        Gamma.Diff dlnP = Gamma.dlnPdf(location.getValue(), theta.getValue(), k.getValue(), value);

        return convertDualNumbersToDiff(dlnP.dPdlocation, dlnP.dPdtheta, dlnP.dPdk, dlnP.dPdx);
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dPdlocation,
                                                             DoubleTensor dPdtheta,
                                                             DoubleTensor dPdk,
                                                             DoubleTensor dPdx) {

        PartialDerivatives dPdInputsFromA = location.getDualNumber().getPartialDerivatives().multiplyBy(dPdlocation);
        PartialDerivatives dPdInputsFromTheta = theta.getDualNumber().getPartialDerivatives().multiplyBy(dPdtheta);
        PartialDerivatives dPdInputsFromK = k.getDualNumber().getPartialDerivatives().multiplyBy(dPdk);
        PartialDerivatives dPdInputs = dPdInputsFromA.add(dPdInputsFromTheta).add(dPdInputsFromK);

        if (!this.isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return dPdInputs.asMap();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return Gamma.sample(getShape(), location.getValue(), theta.getValue(), k.getValue(), random);
    }

}
