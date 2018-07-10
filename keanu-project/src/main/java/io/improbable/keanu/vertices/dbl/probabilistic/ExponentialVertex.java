package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Exponential;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.tensor.Tensor.SCALAR_SHAPE;
import static io.improbable.keanu.tensor.TensorShape.concat;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class ExponentialVertex extends ProbabilisticDouble {

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
    public double logPdf(DoubleTensor value) {

        DoubleTensor locationValues = location.getValue();
        DoubleTensor lambdaValues = lambda.getValue();

        DoubleTensor logPdfs = Exponential.logPdf(locationValues, lambdaValues, value);

        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        Exponential.DiffLogP dlnP = Exponential.dlnPdf(location.getValue(), lambda.getValue(), value);
        return convertDualNumbersToDiff(dlnP.dLogPdlocation, dlnP.dLogPdlambda, dlnP.dLogPdx);
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dLogPdlocation,
                                                             DoubleTensor dLogPdlambda,
                                                             DoubleTensor dLogPdx) {

        PartialDerivatives dLogPdInputsFromA = location.getDualNumber().getPartialDerivatives().multiplyBy(dLogPdlocation);
        PartialDerivatives dLogPdInputsFromB = lambda.getDualNumber().getPartialDerivatives().multiplyBy(dLogPdlambda);
        PartialDerivatives dLogPdInputs = dLogPdInputsFromA.add(dLogPdInputsFromB);

        if (!this.isObserved()) {
            dLogPdInputs.putWithRespectTo(getId(), dLogPdx.reshape(concat(SCALAR_SHAPE, dLogPdx.getShape())));
        }

        PartialDerivatives summed = dLogPdInputs.sum(true, TensorShape.dimensionRange(0, getShape().length));
        return summed.asMap();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return Exponential.sample(getShape(), location.getValue(), lambda.getValue(), random);
    }

}


