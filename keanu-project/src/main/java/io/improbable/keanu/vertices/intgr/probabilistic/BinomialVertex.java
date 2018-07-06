package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.distributions.discrete.Binomial;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

public class BinomialVertex extends ProbabilisticInteger {

    private final DoubleVertex p;
    private final IntegerVertex n;

    public BinomialVertex(int[] tensorShape, DoubleVertex p, IntegerVertex n) {
        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, p.getShape(), n.getShape());
        this.p = p;
        this.n = n;

        setParents(p, n);
        setValue(IntegerTensor.placeHolder(tensorShape));
    }

    public BinomialVertex(int[] tensorShape, double p, IntegerVertex n) {
        this(tensorShape, ConstantVertex.of(p), n);
    }

    public BinomialVertex(int[] tensorShape, DoubleVertex p, int n) {
        this(tensorShape, p, ConstantVertex.of(n));
    }

    public BinomialVertex(int[] tensorShape, double p, int n) {
        this(tensorShape, ConstantVertex.of(p), ConstantVertex.of(n));
    }

    public BinomialVertex(DoubleVertex p, IntegerVertex n) {
        this(checkHasSingleNonScalarShapeOrAllScalar(p.getShape(), n.getShape()), p, n);
    }

    public BinomialVertex(double p, IntegerVertex n) {
        this(ConstantVertex.of(p), n);
    }

    public BinomialVertex(DoubleVertex p, int n) {
        this(p, ConstantVertex.of(n));
    }

    public BinomialVertex(double p, int n) {
        this(ConstantVertex.of(p), ConstantVertex.of(n));
    }

    @Override
    public double logPmf(IntegerTensor kTensor) {
        return Binomial.logPmf(kTensor, p.getValue(), n.getValue()).sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPmf(IntegerTensor value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return Binomial.sample(getShape(), p.getValue(), n.getValue(), random);
    }
}
