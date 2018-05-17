package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.tensors.continuous.TensorExponential;
import io.improbable.keanu.distributions.tensors.continuous.TensorGaussian;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorPartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticVertexShaping.checkParentShapes;
import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticVertexShaping.getShapeProposal;

public class TensorExponentialVertex extends ProbabilisticDoubleTensor {

    private final DoubleTensorVertex a;
    private final DoubleTensorVertex b;
    private final KeanuRandom random;

    /**
     * One a or b or both driving an arbitrarily shaped tensor of Exponential
     *
     * @param shape  the desired shape of the vertex
     * @param a the a of the Exponential with either the same shape as specified for this vertex or a scalar
     * @param b the b of the Exponential with either the same shape as specified for this vertex or a scalar
     * @param random the source of randomness
     */
    public TensorExponentialVertex(int[] shape, DoubleTensorVertex a, DoubleTensorVertex b, KeanuRandom random) {

        checkParentShapes(shape, a.getValue(), b.getValue());

        this.a = a;
        this.b = b;
        this.random = random;
        setParents(a, b);
        setValue(DoubleTensor.placeHolder(shape));
    }

    /**
     * One to one constructor for mapping some shape of a and b to
     * a matching shaped exponential.
     *
     * @param a the a of the Exponential with either the same shape as specified for this vertex or a scalar
     * @param b the b of the Exponential with either the same shape as specified for this vertex or a scalar
     * @param random the source of randomness
     */
    public TensorExponentialVertex(DoubleTensorVertex a, DoubleTensorVertex b, KeanuRandom random) {
        this(getShapeProposal(a.getValue(), b.getValue()), a, b, random);
    }

    public TensorExponentialVertex(DoubleTensorVertex a, double b, KeanuRandom random) {
        this(a, new ConstantTensorVertex(b), random);
    }

    public TensorExponentialVertex(double a, DoubleTensorVertex b, KeanuRandom random) {
        this(new ConstantTensorVertex(a), b, random);
    }

    public TensorExponentialVertex(double a, double b, KeanuRandom random) {
        this(new ConstantTensorVertex(a), new ConstantTensorVertex(b), random);
    }

    public TensorExponentialVertex(DoubleTensorVertex a, DoubleTensorVertex b) {
        this(getShapeProposal(a.getValue(), b.getValue()), a, b, new KeanuRandom());
    }

    public TensorExponentialVertex(DoubleTensorVertex a, double b) {
        this(a, new ConstantTensorVertex(b), new KeanuRandom());
    }

    public TensorExponentialVertex(double a, DoubleTensorVertex b) {
        this(new ConstantTensorVertex(a), b, new KeanuRandom());
    }

    public TensorExponentialVertex(double a, double b) {
        this(new ConstantTensorVertex(a), new ConstantTensorVertex(b), new KeanuRandom());
    }

    @Override
    public double logPdf(DoubleTensor value) {

        DoubleTensor aValues = a.getValue();
        DoubleTensor bValues = b.getValue();

        DoubleTensor logPdfs = TensorExponential.logPdf(aValues, bValues, value);

        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        TensorExponential.Diff dlnP = TensorExponential.dlnPdf(a.getValue(), b.getValue(), value);

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
    public DoubleTensor sample() {
        return TensorExponential.sample(getValue().getShape(), a.getValue(), b.getValue(), random);
    }
}


