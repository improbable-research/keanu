package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorLogistic;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class LogisticVertex extends ProbabilisticDouble {

    private final DoubleVertex a;
    private final DoubleVertex b;

    /**
     * One a or b or both driving an arbitrarily shaped tensor of Logistic
     *
     * @param shape the desired shape of the vertex
     * @param a     the a of the Logistic with either the same shape as specified for this vertex or a scalar
     * @param b     the b of the Logistic with either the same shape as specified for this vertex or a scalar
     */
    public LogisticVertex(int[] shape, DoubleVertex a, DoubleVertex b) {

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
    public LogisticVertex(DoubleVertex a, DoubleVertex b) {
        this(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), a.getShape()), a, b);
    }

    public LogisticVertex(DoubleVertex a, double b) {
        this(a, new ConstantDoubleVertex(b));
    }

    public LogisticVertex(double a, DoubleVertex b) {
        this(new ConstantDoubleVertex(a), b);
    }

    public LogisticVertex(double a, double b) {
        this(new ConstantDoubleVertex(a), new ConstantDoubleVertex(b));
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

        PartialDerivatives dPdInputsFromA = a.getDualNumber().getPartialDerivatives().multiplyBy(dPda);
        PartialDerivatives dPdInputsFromB = b.getDualNumber().getPartialDerivatives().multiplyBy(dPdb);
        PartialDerivatives dPdInputs = dPdInputsFromA.add(dPdInputsFromB);

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
