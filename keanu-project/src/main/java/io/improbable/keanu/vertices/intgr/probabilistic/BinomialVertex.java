package io.improbable.keanu.vertices.intgr.probabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.Map;

import io.improbable.keanu.distributions.discrete.Binomial;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;

public class BinomialVertex extends IntegerVertex implements Probabilistic<IntegerTensor> {

    private final DoubleVertex p;
    private final IntegerVertex n;

    // package private
    public BinomialVertex(int[] tensorShape, DoubleVertex p, IntegerVertex n) {
        super(new ProbabilisticValueUpdater<>(), Observable.observableTypeFor(BinomialVertex.class));

        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, p.getShape(), n.getShape());
        this.p = p;
        this.n = n;

        setParents(p, n);
        setValue(IntegerTensor.placeHolder(tensorShape));
    }

    @Override
    public double logProb(IntegerTensor kTensor) {
        return Binomial.withParameters(p.getValue(), n.getValue()).logProb(kTensor).sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(IntegerTensor value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return Binomial.withParameters(p.getValue(), n.getValue()).sample(getShape(), random);
    }
}
