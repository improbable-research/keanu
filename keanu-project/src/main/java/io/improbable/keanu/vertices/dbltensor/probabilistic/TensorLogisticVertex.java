package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorLogistic;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorPartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.vertices.dbltensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.vertices.dbltensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class TensorLogisticVertex extends ProbabilisticDoubleTensor {

    private final DoubleTensorVertex a;
    private final DoubleTensorVertex b;

    /**
     * One a or b or both driving an arbitrarily shaped tensor of Logistic
     *
     * @param shape  the desired shape of the vertex
     * @param a the a of the Logistic with either the same shape as specified for this vertex or a scalar
     * @param b the b of the Logistic with either the same shape as specified for this vertex or a scalar
     */
    public TensorLogisticVertex(int[] shape, DoubleTensorVertex a, DoubleTensorVertex b) {

        checkTensorsMatchNonScalarShapeOrAreScalar(shape, a.getShape(), b.getShape());

        this.a = a;
        this.b = b;
        setParents(a, b);
        setValue(DoubleTensor.placeHolder(shape));
    }

    /**
     * One to one constructor for mapping some shape of a and b to
     * a matching shaped logistic.
     *
     * @param a the a of the Logistic with either the same shape as specified for this vertex or a scalar
     * @param b the b of the Logistic with either the same shape as specified for this vertex or a scalar
     */
    public TensorLogisticVertex(DoubleTensorVertex a, DoubleTensorVertex b) {
        this(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), a.getShape()), a, b);
    }

    public TensorLogisticVertex(DoubleTensorVertex a, double b) {
        this(a, new ConstantTensorVertex(b));
    }

    public TensorLogisticVertex(double a, DoubleTensorVertex b) {
        this(new ConstantTensorVertex(a), b);
    }

    public TensorLogisticVertex(double a, double b) {
        this(new ConstantTensorVertex(a), new ConstantTensorVertex(b));
    }

    @Override
    public double logPdf(DoubleTensor value) {
        DoubleTensor aValues = a.getValue();
        DoubleTensor bValues = b.getValue();

        DoubleTensor logPdfs = TensorLogistic.logPdf(aValues, bValues, value);

        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        TensorLogistic.Diff dlnP = TensorLogistic.dlnPdf(a.getValue(), b.getValue(), value);
        return convertDualNumbersToDiff(dlnP.dPda, dlnP.dPdb, dlnP.dPdx);
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dPda,
                                                             DoubleTensor dPdb,
                                                             DoubleTensor dPdx) {

        TensorPartialDerivatives dPdInputsFromA = a.getDualNumber().getPartialDerivatives().multiplyBy(dPda);
        TensorPartialDerivatives dPdInputsFromB = b.getDualNumber().getPartialDerivatives().multiplyBy(dPdb);
        TensorPartialDerivatives dPdInputs = dPdInputsFromA.add(dPdInputsFromB);

        if (!this.isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return dPdInputs.asMap();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return TensorLogistic.sample(getShape(), a.getValue(), b.getValue(), random);
    }
}
