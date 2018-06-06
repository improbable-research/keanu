package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.apache.commons.math3.special.Gamma;

public class TensorChiSquared {

    private static final double LOG_TWO = Math.log(2);

    private TensorChiSquared() {
    }

    public static DoubleTensor sample(int[] shape, IntegerTensor k, KeanuRandom random) {
        return random.nextGamma(shape, DoubleTensor.ZERO_SCALAR, DoubleTensor.TWO_SCALAR, k.toDouble().div(2));
    }

    public static DoubleTensor logPdf(IntegerTensor k, DoubleTensor x) {
        DoubleTensor halfK = k.toDouble().div(2);
        DoubleTensor numerator = halfK.minus(1).timesInPlace(x.log()).minusInPlace(x.div(2));
        DoubleTensor denominator = halfK.times(LOG_TWO).plusInPlace(halfK.apply(Gamma::gamma).logInPlace());
        return numerator.minusInPlace(denominator);
    }

}