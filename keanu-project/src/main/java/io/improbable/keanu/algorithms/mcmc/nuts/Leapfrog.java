package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;

import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * Leapfrog performs a movement through physical space with the introduction of a momentum variable.
 * This is required for sampling in Hamiltonian and NUTS.
 */
public class Leapfrog {

    private final Map<VertexId, DoubleTensor> position;
    private final Map<VertexId, DoubleTensor> momentum;
    private final Map<VertexId, DoubleTensor> gradient;

    /**
     * @param position the position of the vertices
     * @param momentum the momentum of the vertices
     * @param gradient the gradient of the vertices
     */
    Leapfrog(Map<VertexId, DoubleTensor> position,
             Map<VertexId, DoubleTensor> momentum,
             Map<VertexId, DoubleTensor> gradient) {
        this.position = position;
        this.momentum = momentum;
        this.gradient = gradient;
    }

    /**
     * Performs one leapfrog of the vertices with a time delta as defined by epsilon
     *
     * @param latentVertices                the latent vertices
     * @param logProbGradientCalculator     the calculator for the log prob gradient
     * @param epsilon                       the time delta

     * @return a new leapfrog having taken one step through space
     */
    public Leapfrog step(final List<Vertex<DoubleTensor>> latentVertices,
                         final LogProbGradientCalculator logProbGradientCalculator,
                         final double epsilon) {

        final double halfTimeStep = epsilon / 2.0;

        Map<VertexId, DoubleTensor> nextMomentum = stepMomentum(halfTimeStep, momentum, gradient);
        Map<VertexId, DoubleTensor> nextPosition = stepPosition(latentVertices, halfTimeStep, nextMomentum, position);

        VertexValuePropagation.cascadeUpdate(latentVertices);
        Map<VertexId, DoubleTensor> nextPositionGradient = logProbGradientCalculator.getJointLogProbGradientWrtLatents();

        nextMomentum = stepMomentum(halfTimeStep, nextMomentum, nextPositionGradient);

        return new Leapfrog(nextPosition, nextMomentum, nextPositionGradient);
    }

    private Map<VertexId, DoubleTensor> stepPosition(List<Vertex<DoubleTensor>> latentVertices, double halfTimeStep, Map<VertexId, DoubleTensor> nextMomentum, Map<VertexId, DoubleTensor> position) {
        Map<VertexId, DoubleTensor> nextPosition = new HashMap<>();
        for (Vertex<DoubleTensor> latent : latentVertices) {
            final DoubleTensor nextPositionForLatent = nextMomentum.get(latent.getId()).
                times(halfTimeStep).
                plusInPlace(
                    position.get(latent.getId())
                );
            nextPosition.put(latent.getId(), nextPositionForLatent);
            latent.setValue(nextPositionForLatent);
        }
        return nextPosition;
    }

    private Map<VertexId, DoubleTensor> stepMomentum(double halfTimeStep, Map<VertexId, DoubleTensor> momentum, Map<VertexId, DoubleTensor> gradient) {
        Map<VertexId, DoubleTensor> nextMomentum = new HashMap<>();
        for (Map.Entry<VertexId, DoubleTensor> rEntry : momentum.entrySet()) {
            final DoubleTensor updatedMomentum = (gradient.get(rEntry.getKey()).times(halfTimeStep)).plusInPlace(rEntry.getValue());
            nextMomentum.put(rEntry.getKey(), updatedMomentum);
        }
        return nextMomentum;
    }

    public double halfDotProductMomentum() {
        return 0.5 * dotProduct(momentum);
    }

    public Map<VertexId, DoubleTensor> getPosition() {
        return position;
    }

    public Map<VertexId, DoubleTensor> getMomentum() {
        return momentum;
    }

    public Map<VertexId, DoubleTensor> getGradient() {
        return gradient;
    }

    public Leapfrog makeJumpTo(Map<VertexId, DoubleTensor> position, Map<VertexId, DoubleTensor> gradient) {
        return new Leapfrog(position, getMomentum(), gradient);
    }

    private static double dotProduct(Map<VertexId, DoubleTensor> momentums) {
        double dotProduct = 0.0;
        for (DoubleTensor momentum : momentums.values()) {
            dotProduct += momentum.pow(2).sum();
        }
        return dotProduct;
    }

}