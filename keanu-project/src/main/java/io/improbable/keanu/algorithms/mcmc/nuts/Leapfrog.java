package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.Getter;

import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * Leapfrog performs a movement through physical space with the introduction of a velocity variable.
 * This is required for sampling in NUTS.
 */
class Leapfrog {

    @Getter
    private final Map<VariableReference, DoubleTensor> position;

    @Getter
    private final Map<VariableReference, DoubleTensor> momentum;

    @Getter
    private final Map<VariableReference, DoubleTensor> velocity;

    @Getter
    private final Map<? extends VariableReference, DoubleTensor> gradient;

    @Getter
    private final double energy;

    @Getter
    private final double logProb;

    @Getter
    private final double kineticEnergy;

    /**
     * @param position the position of the variables
     * @param momentum the velocity of the variables
     * @param gradient the gradient of the variables
     */
    Leapfrog(Map<VariableReference, DoubleTensor> position,
             Map<VariableReference, DoubleTensor> momentum,
             Map<? extends VariableReference, DoubleTensor> gradient,
             double logProb) {

        this.position = position;
        this.momentum = momentum;
        this.velocity = momentum;
        this.gradient = gradient;

        //moment is equal to velocity at the moment because mass is 1
        this.kineticEnergy = 0.5 * dotProduct(velocity);
        this.energy = kineticEnergy - logProb;

        this.logProb = logProb;
    }

    /**
     * Performs one leapfrog of the variables with a time delta as defined by epsilon
     *
     * @param latentVariables           the latent variables
     * @param logProbGradientCalculator the calculator for the log prob gradient
     * @param epsilon                   the time delta
     * @return a new leapfrog having taken one step through space
     */
    public Leapfrog step(final List<? extends Variable<DoubleTensor, ?>> latentVariables,
                         final ProbabilisticModelWithGradient logProbGradientCalculator,
                         final double epsilon) {

        final double halfTimeStep = epsilon / 2.0;

        Map<VariableReference, DoubleTensor> nextMomentum = stepMomentum(halfTimeStep, velocity, gradient);
        Map<VariableReference, DoubleTensor> nextPosition = stepPosition(latentVariables, halfTimeStep, nextMomentum, position);

        Map<? extends VariableReference, DoubleTensor> nextPositionGradient = logProbGradientCalculator.logProbGradients(nextPosition);
        double logProbAtPosition = logProbGradientCalculator.logProb();

        nextMomentum = stepMomentum(halfTimeStep, nextMomentum, nextPositionGradient);

        return new Leapfrog(nextPosition, nextMomentum, nextPositionGradient, logProbAtPosition);
    }

    private static Map<VariableReference, DoubleTensor> stepPosition(List<? extends Variable<DoubleTensor, ?>> latentVariables, double halfTimeStep, Map<VariableReference, DoubleTensor> nextMomentum, Map<? extends VariableReference, DoubleTensor> position) {
        Map<VariableReference, DoubleTensor> nextPosition = new HashMap<>();

        for (Variable<DoubleTensor, ?> latent : latentVariables) {

            final DoubleTensor nextPositionForLatent = nextMomentum.get(latent.getReference()).
                times(halfTimeStep).
                plusInPlace(
                    position.get(latent.getReference())
                );

            nextPosition.put(latent.getReference(), nextPositionForLatent);
        }

        return nextPosition;
    }

    private static Map<VariableReference, DoubleTensor> stepMomentum(double halfTimeStep, Map<? extends VariableReference, DoubleTensor> momentum, Map<? extends VariableReference, DoubleTensor> gradient) {
        Map<VariableReference, DoubleTensor> nextMomentum = new HashMap<>();
        for (Map.Entry<? extends VariableReference, DoubleTensor> rEntry : momentum.entrySet()) {
            final DoubleTensor updatedMomentum = gradient.get(rEntry.getKey()).times(halfTimeStep).plusInPlace(rEntry.getValue());
            nextMomentum.put(rEntry.getKey(), updatedMomentum);
        }
        return nextMomentum;
    }

    private static double dotProduct(Map<? extends VariableReference, DoubleTensor> values) {
        double dotProduct = 0.0;
        for (DoubleTensor value : values.values()) {
            dotProduct += value.times(value).sum();
        }
        return dotProduct;
    }

}