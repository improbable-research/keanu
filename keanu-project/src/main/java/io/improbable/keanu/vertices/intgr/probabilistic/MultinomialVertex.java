package io.improbable.keanu.vertices.intgr.probabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.Map;

import org.apache.commons.lang3.ArrayUtils;

import io.improbable.keanu.distributions.discrete.Multinomial;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;

public class MultinomialVertex extends IntegerVertex implements ProbabilisticInteger {

    private final DoubleVertex p;
    private final IntegerVertex n;

    public MultinomialVertex(int[] tensorShape, IntegerVertex n, DoubleVertex p) {
        super(new ProbabilisticValueUpdater<>());

        int[] tensorShapePrefixedByOne = tensorShape;
        if (tensorShape.length > 2) {
            tensorShapePrefixedByOne = ArrayUtils.insert(0, tensorShape, 1);
        }
        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, n.getShape());
        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, p.slice(0, 0).getShape());

        this.p = p;
        this.n = n;

        setParents(p);
        addParent(n);
        setValue(IntegerTensor.placeHolder(tensorShape));
    }

    public MultinomialVertex(IntegerVertex n, DoubleVertex p) {
        this(n.getShape(), n, p);
    }

    @Override
    public double logProb(IntegerTensor kTensor) {
        return Multinomial.withParameters(n.getValue(), p.getValue()).logProb(kTensor).sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(IntegerTensor value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return Multinomial.withParameters(n.getValue(), p.getValue()).sample(getShape(), random);
    }
}
