package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorBeta;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorPartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticVertexShaping.checkParentShapes;
import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticVertexShaping.getShapeProposal;

public class TensorBetaVertex extends ProbabilisticDoubleTensor {

    private final DoubleTensorVertex alpha;
    private final DoubleTensorVertex beta;

    public TensorBetaVertex(int[] shape, DoubleTensorVertex alpha, DoubleTensorVertex beta) {

        checkParentShapes(shape, alpha.getValue(), beta.getValue());

        this.alpha = alpha;
        this.beta = beta;
        setParents(alpha, beta);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public TensorBetaVertex(DoubleTensorVertex alpha, DoubleTensorVertex beta) {
        this(getShapeProposal(alpha.getValue(), beta.getValue()), alpha, beta);
    }

    public TensorBetaVertex(DoubleTensorVertex alpha, double beta) {
        this(alpha, new ConstantTensorVertex(beta));
    }

    public TensorBetaVertex(double alpha, DoubleTensorVertex beta) {
        this(new ConstantTensorVertex(alpha), beta);
    }

    public TensorBetaVertex(double alpha, double beta) {
        this(new ConstantTensorVertex(alpha), new ConstantTensorVertex(beta));
    }

    @Override
    public double logPdf(DoubleTensor value) {

        DoubleTensor alphaValues = alpha.getValue();
        DoubleTensor betaValues = beta.getValue();

        DoubleTensor logPdfs = TensorBeta.logPdf(alphaValues, betaValues, value);

        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        TensorBeta.Diff dlnP = TensorBeta.dlnPdf(alpha.getValue(), beta.getValue(), value);
        return convertDualNumbersToDiff(dlnP.dPdalpha, dlnP.dPdbeta, dlnP.dPdx);
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dPdalpha,
                                                             DoubleTensor dPdbeta,
                                                             DoubleTensor dPdx) {

        TensorPartialDerivatives dPdInputsFromAlpha = alpha.getDualNumber().getPartialDerivatives().multiplyBy(dPdalpha);
        TensorPartialDerivatives dPdInputsFromBeta = beta.getDualNumber().getPartialDerivatives().multiplyBy(dPdbeta);
        TensorPartialDerivatives dPdInputs = dPdInputsFromAlpha.add(dPdInputsFromBeta);

        if (!this.isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return dPdInputs.asMap();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return TensorBeta.sample(
            getValue().getShape(),
            alpha.getValue(),
            beta.getValue(),
            Nd4jDoubleTensor.scalar(0.),
            Nd4jDoubleTensor.scalar(1.),
            random
        );
    }

}
