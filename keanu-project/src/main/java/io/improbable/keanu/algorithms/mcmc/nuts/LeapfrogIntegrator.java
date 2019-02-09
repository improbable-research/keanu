package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;

import java.util.HashMap;
import java.util.Map;

@AllArgsConstructor
public class LeapfrogIntegrator {

    private final Potential potential;

    /**
     * Performs one leapfrog of the variables with a time delta as defined by epsilon
     *
     * @param logProbGradientCalculator the calculator for the log prob gradient
     * @param epsilon                   the time delta
     * @return a new leapfrog having taken one step through space
     */
    public LeapfrogState step(LeapfrogState fromState, final ProbabilisticModelWithGradient logProbGradientCalculator, final double epsilon) {

        final double halfTimeStep = epsilon / 2.0;

        Map<VariableReference, DoubleTensor> nextMomentum = stepMomentum(halfTimeStep, fromState.getMomentum(), fromState.getGradient());

        Map<VariableReference, DoubleTensor> nextVelocity = potential.getVelocity(nextMomentum);

        Map<VariableReference, DoubleTensor> nextPosition = stepPosition(epsilon, nextVelocity, fromState.getPosition());

        Map<? extends VariableReference, DoubleTensor> nextPositionGradient = logProbGradientCalculator.logProbGradients(nextPosition);
        final double nextPositionLogProb = logProbGradientCalculator.logProb();

        nextMomentum = stepMomentum(halfTimeStep, nextMomentum, nextPositionGradient);

        return new LeapfrogState(nextPosition, nextMomentum, nextPositionGradient, nextPositionLogProb, potential);
    }

    private static Map<VariableReference, DoubleTensor> stepPosition(double dt,
                                                                     Map<VariableReference, DoubleTensor> velocity,
                                                                     Map<VariableReference, DoubleTensor> position) {

        Map<VariableReference, DoubleTensor> nextPosition = new HashMap<>();

        for (VariableReference variableReference : position.keySet()) {

            final DoubleTensor variablePosition = position.get(variableReference);
            final DoubleTensor variableVelocity = velocity.get(variableReference);

            final DoubleTensor nextPositionForLatent = variableVelocity.times(dt).plusInPlace(variablePosition);

            nextPosition.put(variableReference, nextPositionForLatent);
        }

        return nextPosition;
    }

    private static Map<VariableReference, DoubleTensor> stepMomentum(double dt,
                                                                     Map<? extends VariableReference, DoubleTensor> momentum,
                                                                     Map<? extends VariableReference, DoubleTensor> gradient) {
        Map<VariableReference, DoubleTensor> nextMomentum = new HashMap<>();
        for (Map.Entry<? extends VariableReference, DoubleTensor> rEntry : momentum.entrySet()) {
            final DoubleTensor updatedMomentum = gradient.get(rEntry.getKey()).times(dt).plusInPlace(rEntry.getValue());
            nextMomentum.put(rEntry.getKey(), updatedMomentum);
        }
        return nextMomentum;
    }
}
