package io.improbable.keanu.vertices.intgr.probabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import io.improbable.keanu.distributions.discrete.Multinomial;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import java.util.Map;
import java.util.Set;
import org.apache.commons.lang3.ArrayUtils;

public class MultinomialVertex extends IntegerVertex implements ProbabilisticInteger {

    private final DoubleVertex p;
    private final IntegerVertex n;

    public MultinomialVertex(int[] tensorShape, IntegerVertex n, DoubleVertex p) {

        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, n.getShape());
        int[] pShapeExcludingFirstDimension = ArrayUtils.remove(p.getShape(), 0);
        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, pShapeExcludingFirstDimension);

        this.p = p;
        this.n = n;

        setParents(p);
        addParent(n);
        setValue(IntegerTensor.placeHolder(tensorShape));
    }

    public MultinomialVertex(IntegerVertex n, DoubleVertex p) {
        this(n.getShape(), n, p);
    }

    public MultinomialVertex(int n, DoubleVertex p) {
        this(ConstantVertex.of(n), p);
    }

    @Override
    public double logProb(IntegerTensor kTensor) {
        return Multinomial.withParameters(n.getValue(), p.getValue()).logProb(kTensor).sum();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(
            IntegerTensor value, Set<? extends Vertex> withRespectTo) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return Multinomial.withParameters(n.getValue(), p.getValue()).sample(getShape(), random);
    }
}
