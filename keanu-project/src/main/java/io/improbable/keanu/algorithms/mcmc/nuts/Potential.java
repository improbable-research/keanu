package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;

public interface Potential {

    void initialize(Map<VariableReference, DoubleTensor> shapeLike);

    void update(Map<VariableReference, DoubleTensor> position);

    Map<VariableReference, DoubleTensor> randomMomentum(KeanuRandom random);

    Map<VariableReference, DoubleTensor> getVelocity(Map<VariableReference, DoubleTensor> momentum);

    double getKineticEnergy(Map<VariableReference, DoubleTensor> momentum, Map<VariableReference, DoubleTensor> velocity);
}
