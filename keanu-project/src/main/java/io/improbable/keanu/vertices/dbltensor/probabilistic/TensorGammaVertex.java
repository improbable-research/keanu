package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorGamma;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorPartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticVertexShaping.checkParentShapes;
import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticVertexShaping.getShapeProposal;

public class TensorGammaVertex extends ProbabilisticDoubleTensor {

    private final DoubleTensorVertex a;
    private final DoubleTensorVertex theta;
    private final DoubleTensorVertex k;
    private final KeanuRandom random;

    /**
     * One a, theta or k or all three driving an arbitrarily shaped tensor of Gamma
     *
     * @param shape the desired shape of the vertex
     * @param a      the a of the Gamma with either the same shape as specified for this vertex or a scalar
     * @param theta  the theta of the Gamma with either the same shape as specified for this vertex or a scalar
     * @param k      the k of the Gamma with either the same shape as specified for this vertex or a scalar
     * @param random the source of randomness
     */
    public TensorGammaVertex(int[] shape, DoubleTensorVertex a, DoubleTensorVertex theta, DoubleTensorVertex k, KeanuRandom random) {
        checkParentShapes(shape, a.getValue(), theta.getValue(), k.getValue());

        this.a = a;
        this.theta = theta;
        this.k = k;
        this.random = random;
        setParents(a, theta, k);
        setValue(DoubleTensor.placeHolder(shape));
    }

    /**
     * One to one constructor for mapping some shape of a, theta and k to
     * a matching shaped gamma.
     *
     * @param a      the a of the Gamma with either the same shape as specified for this vertex or a scalar
     * @param theta  the theta of the Gamma with either the same shape as specified for this vertex or a scalar
     * @param k      the k of the Gamma with either the same shape as specified for this vertex or a scalar
     * @param random the source of randomness
     */
    public TensorGammaVertex(DoubleTensorVertex a, DoubleTensorVertex theta, DoubleTensorVertex k, KeanuRandom random) {
        this(getShapeProposal(a.getValue(), theta.getValue(), k.getValue()), a, theta, k, random);
    }

    public TensorGammaVertex(DoubleTensorVertex a, DoubleTensorVertex theta, double k, KeanuRandom random) {
        this(a, theta, new ConstantTensorVertex(k), random);
    }

    public TensorGammaVertex(DoubleTensorVertex a, double theta, DoubleTensorVertex k, KeanuRandom random) {
        this(a, new ConstantTensorVertex(theta), k, random);
    }

    public TensorGammaVertex(DoubleTensorVertex a, double theta, double k, KeanuRandom random) {
        this(a, new ConstantTensorVertex(theta), new ConstantTensorVertex(k), random);
    }

    public TensorGammaVertex(double a, DoubleTensorVertex theta, DoubleTensorVertex k, KeanuRandom random) {
        this(new ConstantTensorVertex(a), theta, k, random);
    }

    public TensorGammaVertex(double a, DoubleTensorVertex theta, double k, KeanuRandom random) {
        this(new ConstantTensorVertex(a), theta, new ConstantTensorVertex(k), random);
    }

    public TensorGammaVertex(double a, double theta, double k, KeanuRandom random) {
        this(new ConstantTensorVertex(a), new ConstantTensorVertex(theta), new ConstantTensorVertex(k), random);
    }

    public TensorGammaVertex(DoubleTensorVertex a, DoubleTensorVertex theta, DoubleTensorVertex k) {
        this(getShapeProposal(a.getValue(), theta.getValue(), k.getValue()), a, theta, k, new KeanuRandom());
    }

    public TensorGammaVertex(DoubleTensorVertex a, DoubleTensorVertex theta, double k) {
        this(a, theta, new ConstantTensorVertex(k), new KeanuRandom());
    }

    public TensorGammaVertex(DoubleTensorVertex a, double theta, DoubleTensorVertex k) {
        this(a, new ConstantTensorVertex(theta), k, new KeanuRandom());
    }

    public TensorGammaVertex(DoubleTensorVertex a, double theta, double k) {
        this(a, new ConstantTensorVertex(theta), new ConstantTensorVertex(k), new KeanuRandom());
    }

    public TensorGammaVertex(double a, DoubleTensorVertex theta, DoubleTensorVertex k) {
        this(new ConstantTensorVertex(a), theta, k, new KeanuRandom());
    }

    public TensorGammaVertex(double a, DoubleTensorVertex theta, double k) {
        this(new ConstantTensorVertex(a), theta, new ConstantTensorVertex(k), new KeanuRandom());
    }

    public TensorGammaVertex(double a, double theta, double k) {
        this(new ConstantTensorVertex(a), new ConstantTensorVertex(theta), new ConstantTensorVertex(k), new KeanuRandom());
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

        TensorPartialDerivatives dPdInputsFromA = a.getDualNumber().getPartialDerivatives().multiplyBy(dPda);
        TensorPartialDerivatives dPdInputsFromTheta = theta.getDualNumber().getPartialDerivatives().multiplyBy(dPdtheta);
        TensorPartialDerivatives dPdInputsFromK = k.getDualNumber().getPartialDerivatives().multiplyBy(dPdk);
        TensorPartialDerivatives dPdInputs = dPdInputsFromA.add(dPdInputsFromTheta).add(dPdInputsFromK);

        if (!this.isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return dPdInputs.asMap();
    }

    @Override
    public DoubleTensor sample() {
        return TensorGamma.sample(getValue().getShape(), a.getValue(), theta.getValue(), k.getValue(), random);
    }
}
