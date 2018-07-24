package io.improbable.keanu.vertices.intgr.probabilistic;

import java.util.Map;

import io.improbable.keanu.distributions.continuous.DistributionOfType;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class BinomialVertex extends DistributionBackedIntegerVertex<NumberTensor<?,?>> {

    // TODO: make package private
    public BinomialVertex(int[] tensorShape, DoubleVertex p, IntegerVertex n) {
        super(tensorShape, DistributionOfType::binomial, p, n);
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(IntegerTensor value) {
        throw new UnsupportedOperationException();
    }
}
