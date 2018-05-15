package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorGamma;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorPartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticVertexShaping.checkParentShapes;
import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticVertexShaping.getShapeProposal;

public class TensorGammaVertex extends ProbabilisticDoubleTensor {

    private final DoubleTensorVertex a;
    private final DoubleTensorVertex theta;
    private final DoubleTensorVertex k;
    private final KeanuRandom random;

    public TensorGammaVertex(int[] shape, DoubleTensorVertex a, DoubleTensorVertex theta, DoubleTensorVertex k, KeanuRandom random) {
        checkParentShapes(shape, a.getValue(), theta.getValue(), k.getValue());

        this.a = a;
        this.theta = theta;
        this.k = k;
        this.random = random;
        setParents(a, theta, k);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public TensorGammaVertex(DoubleTensorVertex a, DoubleTensorVertex theta, DoubleTensorVertex k, KeanuRandom random) {
        this(getShapeProposal(a.getValue(), theta.getValue(), k.getValue()), a, theta, k, random);
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
