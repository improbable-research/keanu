package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.InverseGamma;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class InverseGammaVertex extends ProbabilisticDouble {

    private final DoubleVertex alpha;
    private final DoubleVertex beta;

    /**
     * One alpha or beta or both driving an arbitrarily shaped tensor of Inverse Gamma
     *
     * @param tensorShape the desired shape of the vertex
     * @param alpha       the alpha of the Inverse Gamma with either the same shape as specified for this vertex or alpha scalar
     * @param beta        the beta of the Inverse Gamma with either the same shape as specified for this vertex or alpha scalar
     */
    public InverseGammaVertex(int[] tensorShape, DoubleVertex alpha, DoubleVertex beta) {
        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, alpha.getShape(), beta.getShape());

        this.alpha = alpha;
        this.beta = beta;
        setParents(alpha, beta);
        setValue(DoubleTensor.placeHolder(tensorShape));
    }

    /**
     * One to one constructor for mapping some shape of alpha and beta to
     * alpha matching shaped inverse gamma.
     *
     * @param alpha the alpha of the Inverse Gamma with either the same shape as specified for this vertex or alpha scalar
     * @param beta  the beta of the Inverse Gamma with either the same shape as specified for this vertex or alpha scalar
     */
    public InverseGammaVertex(DoubleVertex alpha, DoubleVertex beta) {
        this(checkHasSingleNonScalarShapeOrAllScalar(alpha.getShape(), beta.getShape()), alpha, beta);
    }

    public InverseGammaVertex(DoubleVertex alpha, double beta) {
        this(alpha, new ConstantDoubleVertex(beta));
    }

    public InverseGammaVertex(double alpha, DoubleVertex beta) {
        this(new ConstantDoubleVertex(alpha), beta);
    }

    public InverseGammaVertex(double alpha, double beta) {
        this(new ConstantDoubleVertex(alpha), new ConstantDoubleVertex(beta));
    }

    public InverseGammaVertex(int[] tensorShape, DoubleVertex alpha, double beta) {
        this(tensorShape, alpha, new ConstantDoubleVertex(beta));
    }

    public InverseGammaVertex(int[] tensorShape, double alpha, DoubleVertex beta) {
        this(tensorShape, new ConstantDoubleVertex(alpha), beta);
    }

    public InverseGammaVertex(int[] tensorShape, double alpha, double beta) {
        this(tensorShape, new ConstantDoubleVertex(alpha), new ConstantDoubleVertex(beta));
    }

    @Override
    public double logPdf(DoubleTensor value) {
        DoubleTensor alphaValues = alpha.getValue();
        DoubleTensor betaValues = beta.getValue();

        DoubleTensor logPdfs = InverseGamma.logPdf(alphaValues, betaValues, value);
        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        InverseGamma.Diff dlnP = InverseGamma.dlnPdf(alpha.getValue(), beta.getValue(), value);

        return convertDualNumbersToDiff(dlnP.dPdalpha, dlnP.dPdbeta, dlnP.dPdx);
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dPdalpha,
                                                             DoubleTensor dPdbeta,
                                                             DoubleTensor dPdx) {

        PartialDerivatives dPdInputsFromA = alpha.getDualNumber().getPartialDerivatives().multiplyBy(dPdalpha);
        PartialDerivatives dPdInputsFromB = beta.getDualNumber().getPartialDerivatives().multiplyBy(dPdbeta);
        PartialDerivatives dPdInputs = dPdInputsFromA.add(dPdInputsFromB);

        if (!this.isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return dPdInputs.asMap();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return InverseGamma.sample(getShape(), alpha.getValue(), beta.getValue(), random);
    }

}
