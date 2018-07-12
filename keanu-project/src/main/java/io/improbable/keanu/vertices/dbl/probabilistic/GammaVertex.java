package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Gamma;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShape.shapeToDesiredRankByPrependingOnes;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class GammaVertex extends ProbabilisticDouble {

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
        Gamma.DiffLogP dlnP = Gamma.dlnPdf(location.getValue(), theta.getValue(), k.getValue(), value);

        return convertDualNumbersToDiff(dlnP.dLogPdlocation, dlnP.dLogPdtheta, dlnP.dLogPdk, dlnP.dLogPdx);
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dLogPdlocation,
                                                             DoubleTensor dLogPdtheta,
                                                             DoubleTensor dLogPdk,
                                                             DoubleTensor dLogPdx) {

        PartialDerivatives dLogPdInputsFromA = location.getDualNumber().getPartialDerivatives().multiplyBy(dLogPdlocation);
        PartialDerivatives dLogPdInputsFromTheta = theta.getDualNumber().getPartialDerivatives().multiplyBy(dLogPdtheta);
        PartialDerivatives dLogPdInputsFromK = k.getDualNumber().getPartialDerivatives().multiplyBy(dLogPdk);
        PartialDerivatives dLogPdInputs = dLogPdInputsFromA.add(dLogPdInputsFromTheta).add(dLogPdInputsFromK);

        if (!this.isObserved()) {
            dLogPdInputs.putWithRespectTo(getId(), dLogPdx.reshape(
                shapeToDesiredRankByPrependingOnes(dLogPdx.getShape(), dLogPdx.getRank() + getValue().getRank()))
            );
        }

        PartialDerivatives summed = dLogPdInputs.sum(true, TensorShape.dimensionRange(0, getShape().length));
        return summed.asMap();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return Gamma.sample(getShape(), location.getValue(), theta.getValue(), k.getValue(), random);
    }

}
