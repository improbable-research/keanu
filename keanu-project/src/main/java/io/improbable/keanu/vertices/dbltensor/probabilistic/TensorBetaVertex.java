package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorBeta;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorPartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class TensorBetaVertex extends ProbabilisticDoubleTensor {

    private final DoubleTensorVertex alpha;
    private final DoubleTensorVertex beta;

    /**
     * One alpha or beta or both driving an arbitrarily shaped tensor of Beta
     *
     * @param shape the desired shape of the vertex
     * @param alpha the alpha of the Beta with either the same shape as specified for this vertex or a scalar
     * @param beta  the beta of the Beta with either the same shape as specified for this vertex or a scalar
     */
    public TensorBetaVertex(int[] shape, DoubleTensorVertex alpha, DoubleTensorVertex beta) {

        checkTensorsMatchNonScalarShapeOrAreScalar(shape, alpha.getShape(), beta.getShape());

        this.alpha = alpha;
        this.beta = beta;
        setParents(alpha, beta);
        setValue(DoubleTensor.placeHolder(shape));
    }

    /**
     * One to one constructor for mapping some shape of alpha and beta to
     * a matching shaped Beta.
     *
     * @param alpha the alpha of the Beta with either the same shape as specified for this vertex or a scalar
     * @param beta  the beta of the Beta with either the same shape as specified for this vertex or a scalar
     */
    public TensorBetaVertex(DoubleTensorVertex alpha, DoubleTensorVertex beta) {
        this(checkHasSingleNonScalarShapeOrAllScalar(alpha.getShape(), beta.getShape()), alpha, beta);
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

    public TensorBetaVertex(int[] shape, DoubleTensorVertex alpha, double beta) {
        this(shape, alpha, new ConstantTensorVertex(beta));
    }

    public TensorBetaVertex(int[] shape, double alpha, DoubleTensorVertex beta) {
        this(shape, new ConstantTensorVertex(alpha), beta);
    }

    public TensorBetaVertex(int[] shape, double alpha, double beta) {
        this(shape, new ConstantTensorVertex(alpha), new ConstantTensorVertex(beta));
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
            getShape(),
            alpha.getValue(),
            beta.getValue(),
            Nd4jDoubleTensor.scalar(0.),
            Nd4jDoubleTensor.scalar(1.),
            random
        );
    }

}
