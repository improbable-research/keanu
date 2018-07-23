package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.ParameterName.A;
import static io.improbable.keanu.distributions.dual.ParameterName.B;
import static io.improbable.keanu.distributions.dual.ParameterName.X;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.Map;

import io.improbable.keanu.distributions.continuous.InverseGamma;
import io.improbable.keanu.distributions.dual.ParameterMap;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;

public class InverseGammaVertex extends DoubleVertex implements Probabilistic<DoubleTensor> {

    private final DoubleVertex alpha;
    private final DoubleVertex beta;

    /**
     * One alpha or beta or both driving an arbitrarily shaped tensor of Inverse Gamma
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the vertex
     * @param alpha       the alpha of the Inverse Gamma with either the same shape as specified for this vertex or alpha scalar
     * @param beta        the beta of the Inverse Gamma with either the same shape as specified for this vertex or alpha scalar
     */
    //package private
    InverseGammaVertex(int[] tensorShape, DoubleVertex alpha, DoubleVertex beta) {
        super(new ProbabilisticValueUpdater<>(), Observable.observableTypeFor(InverseGammaVertex.class));

        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, alpha.getShape(), beta.getShape());

        this.alpha = alpha;
        this.beta = beta;
        setParents(alpha, beta);
        setValue(DoubleTensor.placeHolder(tensorShape));
    }

    @Override
    public double logProb(DoubleTensor value) {
        DoubleTensor alphaValues = alpha.getValue();
        DoubleTensor betaValues = beta.getValue();

        DoubleTensor logPdfs = InverseGamma.withParameters(alphaValues, betaValues).logProb(value);
        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {
        ParameterMap<DoubleTensor> dlnP = InverseGamma.withParameters(alpha.getValue(), beta.getValue()).dLogProb(value);

        return convertDualNumbersToDiff(dlnP.get(A).getValue(), dlnP.get(B).getValue(), dlnP.get(X).getValue());

    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dPdalpha,
                                                             DoubleTensor dPdbeta,
                                                             DoubleTensor dPdx) {

        Differentiator differentiator = new Differentiator();
        PartialDerivatives dPdInputsFromA = differentiator.calculateDual((Differentiable) alpha).getPartialDerivatives().multiplyBy(dPdalpha);
        PartialDerivatives dPdInputsFromB = differentiator.calculateDual((Differentiable) beta).getPartialDerivatives().multiplyBy(dPdbeta);
        PartialDerivatives dPdInputs = dPdInputsFromA.add(dPdInputsFromB);

        if (!this.isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return dPdInputs.asMap();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return InverseGamma.withParameters(alpha.getValue(), beta.getValue()).sample(getShape(), random);
    }

}
