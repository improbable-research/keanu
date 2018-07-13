package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.List;
import java.util.Map;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.continuous.Beta;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;

public class BetaVertex extends DoubleVertex implements Probabilistic<DoubleTensor> {

    private final DoubleVertex alpha;
    private final DoubleVertex beta;

    /**
     * One alpha or beta or both that match a proposed tensor shape of Beta.
     *
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the tensor contained in the vertex
     * @param alpha       the alpha of the Beta with either the same tensorShape as specified for this vertex or a scalar
     * @param beta        the beta of the Beta with either the same tensorShape as specified for this vertex or a scalar
     */
    public BetaVertex(int[] tensorShape, DoubleVertex alpha, DoubleVertex beta) {
        super(new ProbabilisticValueUpdater<>(), Observable.observableTypeFor(BetaVertex.class));

        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, alpha.getShape(), beta.getShape());

        this.alpha = alpha;
        this.beta = beta;
        setParents(alpha, beta);
        setValue(DoubleTensor.placeHolder(tensorShape));
    }

    ContinuousDistribution distribution() {
        return Beta.withParameters(alpha.getValue(), beta.getValue(), DoubleTensor.scalar(0.), DoubleTensor.scalar(1.));
    }
    /**
     * One to one constructor for mapping some tensorShape of alpha and beta to
     * a matching tensorShaped Beta.
     *
     * @param alpha the alpha of the Beta with either the same tensorShape as specified for this vertex or a scalar
     * @param beta  the beta of the Beta with either the same tensorShape as specified for this vertex or a scalar
     */
    public BetaVertex(DoubleVertex alpha, DoubleVertex beta) {
        this(checkHasSingleNonScalarShapeOrAllScalar(alpha.getShape(), beta.getShape()), alpha, beta);
    }

    public BetaVertex(DoubleVertex alpha, double beta) {
        this(alpha, new ConstantDoubleVertex(beta));
    }

    public BetaVertex(double alpha, DoubleVertex beta) {
        this(new ConstantDoubleVertex(alpha), beta);
    }

    public BetaVertex(double alpha, double beta) {
        this(new ConstantDoubleVertex(alpha), new ConstantDoubleVertex(beta));
    }

    public BetaVertex(int[] tensorShape, DoubleVertex alpha, double beta) {
        this(tensorShape, alpha, new ConstantDoubleVertex(beta));
    }

    public BetaVertex(int[] tensorShape, double alpha, DoubleVertex beta) {
        this(tensorShape, new ConstantDoubleVertex(alpha), beta);
    }

    public BetaVertex(int[] tensorShape, double alpha, double beta) {
        this(tensorShape, new ConstantDoubleVertex(alpha), new ConstantDoubleVertex(beta));
    }

    @Override
    public double logProb(DoubleTensor value) {
        DoubleTensor logPdfs = distribution().logProb(value);
        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {
        List<DoubleTensor> dlnP = distribution().dLogProb(value);
        return convertDualNumbersToDiff(dlnP.get(0), dlnP.get(1), dlnP.get(2));
    }

    private Map<Long,DoubleTensor> convertDualNumbersToDiff(DoubleTensor dPdalpha, DoubleTensor dPdbeta, DoubleTensor dPdx) {
            Differentiator differentiator = new Differentiator();
            PartialDerivatives dPdInputsFromAlpha = differentiator.calculateDual((Differentiable) alpha).getPartialDerivatives().multiplyBy(dPdalpha);
            PartialDerivatives dPdInputsFromBeta = differentiator.calculateDual((Differentiable) beta).getPartialDerivatives().multiplyBy(dPdbeta);
        PartialDerivatives dPdInputs = dPdInputsFromAlpha.add(dPdInputsFromBeta);

        if (!this.isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return dPdInputs.asMap();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return distribution().sample(getShape(), random);
    }

}
