package io.improbable.keanu.distributions;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.apache.commons.lang3.NotImplementedException;

public interface BaseDistribution {
    default DoubleTensor computeKLDivergence(BaseDistribution q) {
        throw new NotImplementedException("computeKLDivergence is not implemented for this distribution");
    }
}
