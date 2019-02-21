package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import lombok.Value;

import java.util.Map;


/**
 * Leapfrog performs a movement through physical space with the introduction of a velocity variable.
 * This is required for sampling in NUTS.
 */
@Value
@AllArgsConstructor
public class LeapfrogState {

    private final Map<VariableReference, DoubleTensor> position;

    private final Map<VariableReference, DoubleTensor> momentum;

    private final Map<VariableReference, DoubleTensor> velocity;

    private final Map<? extends VariableReference, DoubleTensor> gradient;

    private final double kineticEnergy;

    private final double logProb;

    private final double energy;

    /**
     * @param position  the position of the variables
     * @param momentum  the velocity of the variables
     * @param gradient  the gradient of the variables
     * @param logProb   the log probability at the position
     * @param potential the potential to use for calculating velocity and kinetic energy
     */
    public LeapfrogState(Map<VariableReference, DoubleTensor> position,
                         Map<VariableReference, DoubleTensor> momentum,
                         Map<? extends VariableReference, DoubleTensor> gradient,
                         double logProb,
                         Potential potential) {

        this.position = position;
        this.momentum = momentum;
        this.velocity = potential.getVelocity(momentum);
        this.gradient = gradient;
        this.kineticEnergy = potential.getKineticEnergy(momentum, velocity);
        this.energy = kineticEnergy - logProb;
        this.logProb = logProb;
    }


}