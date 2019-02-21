package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.Value;

import java.util.Map;

@Value
public class Proposal {

    private final Map<VariableReference, DoubleTensor> position;
    private final Map<? extends VariableReference, DoubleTensor> gradient;
    private final Map<VariableReference, ?> sample;
    private final double logProb;

}
