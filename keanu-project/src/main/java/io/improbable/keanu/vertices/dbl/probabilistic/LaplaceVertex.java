package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.Diffs.BETA;
import static io.improbable.keanu.distributions.dual.Diffs.MU;
import static io.improbable.keanu.distributions.dual.Diffs.X;
import static io.improbable.keanu.tensor.TensorShape.shapeToDesiredRankByPrependingOnes;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.Map;

import io.improbable.keanu.distributions.continuous.Laplace;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class LaplaceVertex extends DoubleVertex implements ProbabilisticDouble {

    private final DoubleVertex mu;
    private final DoubleVertex beta;

    /**
     * One mu or beta or both that match a proposed tensor shape of Laplace
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the tensor within the vertex
     * @param mu          the mu of the Laplace with either the same shape as specified for this vertex or a scalar
     * @param beta        the beta of the Laplace with either the same shape as specified for this vertex or a scalar
     */
    public LaplaceVertex(int[] tensorShape, DoubleVertex mu, DoubleVertex beta) {
        super();

        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, mu.getShape(), beta.getShape());

        this.mu = mu;
        this.beta = beta;
        setParents(mu, beta);
        setValue(DoubleTensor.placeHolder(tensorShape));
    }

    /**
     * One to one constructor for mapping some shape of mu and sigma to
     * a matching shaped Laplace.
     *
     * @param mu   the mu of the Laplace with either the same shape as specified for this vertex or a scalar
     * @param beta the beta of the Laplace with either the same shape as specified for this vertex or a scalar
     */
    public LaplaceVertex(DoubleVertex mu, DoubleVertex beta) {
        this(checkHasSingleNonScalarShapeOrAllScalar(mu.getShape(), beta.getShape()), mu, beta);
    }

    public LaplaceVertex(DoubleVertex mu, double beta) {
        this(mu, new ConstantDoubleVertex(beta));
    }

    public LaplaceVertex(double mu, DoubleVertex beta) {
        this(new ConstantDoubleVertex(mu), beta);
    }

    public LaplaceVertex(double mu, double beta) {
        this(new ConstantDoubleVertex(mu), new ConstantDoubleVertex(beta));
    }

    @Override
    public double logProb(DoubleTensor value) {

        DoubleTensor muValues = mu.getValue();
        DoubleTensor betaValues = beta.getValue();

        DoubleTensor logPdfs = Laplace.withParameters(muValues, betaValues).logProb(value);

        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {
        Diffs dlnP = Laplace.withParameters(mu.getValue(), beta.getValue()).dLogProb(value);
        return convertDualNumbersToDiff(dlnP.get(MU).getValue(), dlnP.get(BETA).getValue(), dlnP.get(X).getValue());
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dLogPdmu,
                                                             DoubleTensor dLogPdbeta,
                                                             DoubleTensor dLogPdx) {

        PartialDerivatives dLogPdInputsFromMu = mu.getDualNumber().getPartialDerivatives().multiplyBy(dLogPdmu);
        PartialDerivatives dLogPdInputsFromBeta = beta.getDualNumber().getPartialDerivatives().multiplyBy(dLogPdbeta);
        PartialDerivatives dLogPdInputs = dLogPdInputsFromMu.add(dLogPdInputsFromBeta);

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
        return Laplace.withParameters(mu.getValue(), beta.getValue()).sample(getShape(), random);
    }

}
