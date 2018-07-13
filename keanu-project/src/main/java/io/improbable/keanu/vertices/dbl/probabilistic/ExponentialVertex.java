package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.List;
import java.util.Map;

import io.improbable.keanu.distributions.continuous.Exponential;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;

public class ExponentialVertex extends DoubleVertex implements Probabilistic<DoubleTensor> {

    private final DoubleVertex location;
    private final DoubleVertex lambda;

    /**
     * One location or lambda or both driving an arbitrarily shaped tensor of Exponential
     * <p>
     * pdf = lambda * exp(-lambda*x)
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the vertex
     * @param location    the horizontal shift of the Exponential with either the same shape as specified
     *                    for this vertex or location scalar
     * @param lambda      the lambda of the Exponential with either the same shape as specified for this
     *                    vertex or location scalar.
     */
    public ExponentialVertex(int[] tensorShape, DoubleVertex location, DoubleVertex lambda) {
        super(new ProbabilisticValueUpdater<>(), Observable.observableTypeFor(ExponentialVertex.class));

        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, location.getShape(), lambda.getShape());

        this.location = location;
        this.lambda = lambda;
        setParents(location, lambda);
        setValue(DoubleTensor.placeHolder(tensorShape));
    }

    /**
     * One to one constructor for mapping some shape of location and lambda to
     * location matching shaped exponential.
     *
     * @param location the location of the Exponential with either the same shape as specified for this vertex or location scalar
     * @param lambda   the lambda of the Exponential with either the same shape as specified for this vertex or location scalar
     */
    public ExponentialVertex(DoubleVertex location, DoubleVertex lambda) {
        this(checkHasSingleNonScalarShapeOrAllScalar(location.getShape(), lambda.getShape()), location, lambda);
    }

    public ExponentialVertex(DoubleVertex location, double lambda) {
        this(location, new ConstantDoubleVertex(lambda));
    }

    public ExponentialVertex(double location, DoubleVertex lambda) {
        this(new ConstantDoubleVertex(location), lambda);
    }

    public ExponentialVertex(double location, double lambda) {
        this(new ConstantDoubleVertex(location), new ConstantDoubleVertex(lambda));
    }

    @Override
    public double logProb(DoubleTensor value) {

        DoubleTensor locationValues = location.getValue();
        DoubleTensor lambdaValues = lambda.getValue();

        DoubleTensor logPdfs = Exponential.withParameters(locationValues, lambdaValues).logProb(value);

        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {
        List<DoubleTensor> dlnP = Exponential.withParameters(location.getValue(), lambda.getValue()).dLogProb(value);
        return convertDualNumbersToDiff(dlnP.get(0), dlnP.get(1), dlnP.get(2));
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dPdlocation,
                                                             DoubleTensor dPdlambda,
                                                             DoubleTensor dPdx) {

        Differentiator differentiator = new Differentiator();
        PartialDerivatives dPdInputsFromA = differentiator.calculateDual((Differentiable) location).getPartialDerivatives().multiplyBy(dPdlocation);
        PartialDerivatives dPdInputsFromB = differentiator.calculateDual((Differentiable) lambda).getPartialDerivatives().multiplyBy(dPdlambda);
        PartialDerivatives dPdInputs = dPdInputsFromA.add(dPdInputsFromB);

        if (!this.isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return dPdInputs.asMap();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return Exponential.withParameters(location.getValue(), lambda.getValue()).sample(getShape(), random);
    }

}


