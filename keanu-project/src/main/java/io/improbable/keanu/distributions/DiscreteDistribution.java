package io.improbable.keanu.distributions;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

import java.util.Collections;

public interface DiscreteDistribution extends Distribution<IntegerTensor> {
    @Override
    default DoubleTensor computeKLDivergence(BaseDistribution q) {
        Support<IntegerTensor> pSupport = this.getSupport();
        Support<IntegerTensor> qSupport = ((DiscreteDistribution) q).getSupport();

        checkTensorsMatchNonScalarShapeOrAreScalar(pSupport.getShape(), qSupport.getShape());
        if (!pSupport.isSubsetOf(qSupport)) {
            throw new IllegalArgumentException("q has to have greater or equal support than p");
        }

        int max = Collections.max(pSupport.getMax().minus(pSupport.getMin()).asFlatList());
        DoubleTensor sum = Nd4jDoubleTensor.zeros(pSupport.getShape());
        for (int i = 0; i < max; i++) {
            IntegerTensor t = this.getSupport().getMin().plus(i);

            DoubleTensor pLogPmf = this.logProb(t);
            DoubleTensor qLogPmf = ((DiscreteDistribution) q).logProb(t);

            DoubleTensor pPmf = pLogPmf.exp();

            DoubleTensor result = pPmf.times(pLogPmf.minus(qLogPmf));
            DoubleTensor mask = t.getGreaterThanMask(this.getSupport().getMax()).toDouble();
            result.setWithMaskInPlace(mask, 0.0);
            sum.plusInPlace(result);
        }

        return sum;
    }
}
