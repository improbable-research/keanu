package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * Leapfrog performs a movement through physical space with the introduction of a momentum variable.
 * This is required for sampling in NUTS.
 */
class Leapfrog {

    private final Map<VariableReference, DoubleTensor> position;
    private final Map<VariableReference, DoubleTensor> momentum;
    private final Map<? extends VariableReference, DoubleTensor> gradient;

    /**
     * @param position the position of the variables
     * @param momentum the momentum of the variables
     * @param gradient the gradient of the variables
     */
    Leapfrog(Map<VariableReference, DoubleTensor> position,
             Map<VariableReference, DoubleTensor> momentum,
             Map<? extends VariableReference, DoubleTensor> gradient) {
        this.position = position;
        this.momentum = momentum;
        this.gradient = gradient;
    }

    /**
     * Performs one leapfrog of the variables with a time delta as defined by epsilon
     *
     * @param latentVariables                the latent variables
     * @param logProbGradientCalculator     the calculator for the log prob gradient
     * @param epsilon                       the time delta

     * @return a new leapfrog having taken one step through space
     */
    public Leapfrog step(final List<? extends Variable<DoubleTensor, ?>> latentVariables,
                         final ProbabilisticModelWithGradient logProbGradientCalculator,
                         final double epsilon) {

        final double halfTimeStep = epsilon / 2.0;

        Map<VariableReference, DoubleTensor> nextMomentum = stepMomentum(halfTimeStep, momentum, gradient);
        Map<VariableReference, DoubleTensor> nextPosition = stepPosition(latentVariables, epsilon, nextMomentum, position);

        Map<? extends VariableReference, DoubleTensor> nextPositionGradient = logProbGradientCalculator.logProbGradients(nextPosition);

        nextMomentum = stepMomentum(halfTimeStep, nextMomentum, nextPositionGradient);

        return new Leapfrog(nextPosition, nextMomentum, nextPositionGradient);
    }

    private Map<VariableReference, DoubleTensor> stepPosition(List<? extends Variable<DoubleTensor, ?>> latentVariables, double dt, Map<VariableReference, DoubleTensor> nextMomentum, Map<? extends VariableReference, DoubleTensor> position) {
        Map<VariableReference, DoubleTensor> nextPosition = new HashMap<>();

        for (Variable<DoubleTensor, ?> latent : latentVariables) {

            final DoubleTensor nextPositionForLatent = nextMomentum.get(latent.getReference())
                .times(dt)
                .plusInPlace(position.get(latent.getReference()));

            nextPosition.put(latent.getReference(), nextPositionForLatent);
        }

        return nextPosition;
    }

    private Map<VariableReference, DoubleTensor> stepMomentum(double halfTimeStep, Map<? extends VariableReference, DoubleTensor> momentum, Map<? extends VariableReference, DoubleTensor> gradient) {
        Map<VariableReference, DoubleTensor> nextMomentum = new HashMap<>();
        for (Map.Entry<? extends VariableReference, DoubleTensor> rEntry : momentum.entrySet()) {
            final DoubleTensor updatedMomentum = gradient.get(rEntry.getKey()).times(halfTimeStep).plusInPlace(rEntry.getValue());
            nextMomentum.put(rEntry.getKey(), updatedMomentum);
        }
        return nextMomentum;
    }

    public double halfDotProductMomentum() {
        return 0.5 * dotProduct(momentum);
    }

    public Map<VariableReference, DoubleTensor> getPosition() {
        return position;
    }

    public Map<VariableReference, DoubleTensor> getMomentum() {
        return momentum;
    }

    public Map<? extends VariableReference, DoubleTensor> getGradient() {
        return gradient;
    }

    public Leapfrog makeJumpTo(Map<VariableReference, DoubleTensor> position, Map<? extends VariableReference, DoubleTensor> gradient) {
        return new Leapfrog(position, getMomentum(), gradient);
    }

    private static double dotProduct(Map<? extends VariableReference, DoubleTensor> momentums) {
        double dotProduct = 0.0;
        for (DoubleTensor momentum : momentums.values()) {
            dotProduct += momentum.pow(2).sum();
        }
        return dotProduct;
    }

}