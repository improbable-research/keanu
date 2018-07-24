package io.improbable.keanu.vertices.intgr.probabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.Map;

import io.improbable.keanu.distributions.continuous.DistributionOfType;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class BinomialVertex extends DistributionBackedIntegerVertex<Vertex<? extends NumberTensor<?,?>>, NumberTensor<?,?>> implements Probabilistic<IntegerTensor> {

    private final DoubleVertex p;
    private final IntegerVertex n;

    // package private
    public BinomialVertex(int[] tensorShape, DoubleVertex p, IntegerVertex n) {
        super(tensorShape, DistributionOfType::binomial, p, n);

        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, p.getShape(), n.getShape());
        this.p = p;
        this.n = n;

        setParents(p, n);
        setValue(IntegerTensor.placeHolder(tensorShape));
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(IntegerTensor value) {
        throw new UnsupportedOperationException();
    }
}
