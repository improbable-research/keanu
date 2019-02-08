package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;

public interface Potential {

    void update(Map<VariableReference, DoubleTensor> position, Map<? extends VariableReference, DoubleTensor> gradient, int sampleNum);

    Map<VariableReference, DoubleTensor> random();

    Map<VariableReference, DoubleTensor> getVelocity(Map<VariableReference, DoubleTensor> momentum);

    double getKineticEnergy(Map<VariableReference, DoubleTensor> momentum, Map<VariableReference, DoubleTensor> velocity);
}
