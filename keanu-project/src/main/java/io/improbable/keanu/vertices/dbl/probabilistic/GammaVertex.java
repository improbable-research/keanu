package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorGamma;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class GammaVertex extends ProbabilisticDouble {

    private final DoubleVertex a;
    private final DoubleVertex theta;
    private final DoubleVertex k;

    /**
     * One a, theta or k or all three driving an arbitrarily shaped tensor of Gamma
     *
     * @param shape the desired shape of the vertex
     * @param a     the a of the Gamma with either the same shape as specified for this vertex or a scalar
     * @param theta the theta of the Gamma with either the same shape as specified for this vertex or a scalar
     * @param k     the k of the Gamma with either the same shape as specified for this vertex or a scalar
     */
    public GammaVertex(int[] shape, DoubleVertex a, DoubleVertex theta, DoubleVertex k) {
        checkTensorsMatchNonScalarShapeOrAreScalar(shape, a.getShape(), theta.getShape(), k.getShape());

        this.a = a;
        this.theta = theta;
        this.k = k;
        setParents(a, theta, k);
        setValue(DoubleTensor.placeHolder(shape));
    }

    /**
     * One to one constructor for mapping some shape of a, theta and k to
     * a matching shaped gamma.
     *
     * @param a     the a of the Gamma with either the same shape as specified for this vertex or a scalar
     * @param theta the theta of the Gamma with either the same shape as specified for this vertex or a scalar
     * @param k     the k of the Gamma with either the same shape as specified for this vertex or a scalar
     */
    public GammaVertex(DoubleVertex a, DoubleVertex theta, DoubleVertex k) {
        this(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), theta.getShape(), k.getShape()), a, theta, k);
    }

    public GammaVertex(DoubleVertex a, DoubleVertex theta, double k) {
        this(a, theta, new ConstantDoubleVertex(k));
    }

    public GammaVertex(DoubleVertex a, double theta, DoubleVertex k) {
        this(a, new ConstantDoubleVertex(theta), k);
    }

    public GammaVertex(DoubleVertex a, double theta, double k) {
        this(a, new ConstantDoubleVertex(theta), new ConstantDoubleVertex(k));
    }

    public GammaVertex(double a, DoubleVertex theta, DoubleVertex k) {
        this(new ConstantDoubleVertex(a), theta, k);
    }

    public GammaVertex(double a, DoubleVertex theta, double k) {
        this(new ConstantDoubleVertex(a), theta, new ConstantDoubleVertex(k));
    }

    public GammaVertex(double a, double theta, DoubleVertex k) {
        this(new ConstantDoubleVertex(a), new ConstantDoubleVertex(theta), k);
    }

    public GammaVertex(double a, double theta, double k) {
        this(new ConstantDoubleVertex(a), new ConstantDoubleVertex(theta), new ConstantDoubleVertex(k));
    }

    @Override
    public double logPdf(DoubleTensor value) {
        DoubleTensor aValues = a.getValue();
        DoubleTensor thetaValues = theta.getValue();
        DoubleTensor kValues = k.getValue();

        DoubleTensor logPdfs = TensorGamma.logPdf(aValues, thetaValues, kValues, value);
        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        TensorGamma.Diff dlnP = TensorGamma.dlnPdf(a.getValue(), theta.getValue(), k.getValue(), value);

        return convertDualNumbersToDiff(dlnP.dPda, dlnP.dPdtheta, dlnP.dPdk, dlnP.dPdx);
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dPda,
                                                             DoubleTensor dPdtheta,
                                                             DoubleTensor dPdk,
                                                             DoubleTensor dPdx) {

        PartialDerivatives dPdInputsFromA = a.getDualNumber().getPartialDerivatives().multiplyBy(dPda);
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
        return TensorGamma.sample(getShape(), a.getValue(), theta.getValue(), k.getValue(), random);
    }

}
