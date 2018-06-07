package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorInverseGamma;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class InverseGammaVertex extends ProbabilisticDouble {

    private final DoubleVertex a;
    private final DoubleVertex b;

    /**
     * One a or b or both driving an arbitrarily shaped tensor of Inverse Gamma
     *
     * @param shape the desired shape of the vertex
     * @param a     the a of the Inverse Gamma with either the same shape as specified for this vertex or a scalar
     * @param b     the b of the Inverse Gamma with either the same shape as specified for this vertex or a scalar
     */
    public InverseGammaVertex(int[] shape, DoubleVertex a, DoubleVertex b) {
        checkTensorsMatchNonScalarShapeOrAreScalar(shape, a.getShape(), b.getShape());

        this.a = a;
        this.b = b;
        setParents(a, b);
        setValue(DoubleTensor.placeHolder(shape));
    }

    /**
     * One to one constructor for mapping some shape of a and b to
     * a matching shaped inverse gamma.
     *
     * @param a the a of the Inverse Gamma with either the same shape as specified for this vertex or a scalar
     * @param b the b of the Inverse Gamma with either the same shape as specified for this vertex or a scalar
     */
    public InverseGammaVertex(DoubleVertex a, DoubleVertex b) {
        this(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), b.getShape()), a, b);
    }

    public InverseGammaVertex(DoubleVertex a, double b) {
        this(a, new ConstantDoubleVertex(b));
    }

    public InverseGammaVertex(double a, DoubleVertex b) {
        this(new ConstantDoubleVertex(a), b);
    }

    public InverseGammaVertex(double a, double b) {
        this(new ConstantDoubleVertex(a), new ConstantDoubleVertex(b));
    }

    public InverseGammaVertex(int[] shape, DoubleVertex a, double b) {
        this(shape, a, new ConstantDoubleVertex(b));
    }

    public InverseGammaVertex(int[] shape, double a, DoubleVertex b) {
        this(shape, new ConstantDoubleVertex(a), b);
    }

    public InverseGammaVertex(int[] shape, double a, double b) {
        this(shape, new ConstantDoubleVertex(a), new ConstantDoubleVertex(b));
    }

    @Override
    public double logPdf(DoubleTensor value) {
        DoubleTensor aValues = a.getValue();
        DoubleTensor bValues = b.getValue();

        DoubleTensor logPdfs = TensorInverseGamma.logPdf(aValues, bValues, value);
        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        TensorInverseGamma.Diff dlnP = TensorInverseGamma.dlnPdf(a.getValue(), b.getValue(), value);

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
        return TensorInverseGamma.sample(getShape(), a.getValue(), b.getValue(), random);
    }

}
