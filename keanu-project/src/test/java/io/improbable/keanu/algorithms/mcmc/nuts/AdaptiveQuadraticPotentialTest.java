package io.improbable.keanu.algorithms.mcmc.nuts;

import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Test;

import java.util.Map;

import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.ones;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.zeros;
import static org.junit.Assert.assertEquals;

public class AdaptiveQuadraticPotentialTest {


    @Test
    public void doesQuadraticPotentialBeforeAnyUpdate() {
        DoubleVertex v = new GaussianVertex(0, 1);
        v.setValue(1);

        Map<VariableReference, DoubleTensor> position = ImmutableMap.of(v.getReference(), v.getValue());

        AdaptiveQuadraticPotential potential = new AdaptiveQuadraticPotential(
            zeros(position),
            ones(position),
            10,
            10,
            101,
            KeanuRandom.getDefaultRandom()
        );

        Map<VariableReference, DoubleTensor> momentum = ImmutableMap.of(v.getReference(), DoubleTensor.scalar(0.5));
        Map<VariableReference, DoubleTensor> velocity = potential.getVelocity(momentum);

        double kineticEnergy = potential.getKineticEnergy(momentum, velocity);

        assertEquals(0.5, velocity.get(v.getReference()).scalar(), 1e-6);
        assertEquals(0.5 * Math.pow(0.5, 2), kineticEnergy, 1e-6);
    }

    @Test
    public void doesUpdateAfterAdaptSample() {
        DoubleVertex v = new GaussianVertex(0, 1);
        v.setValue(1.0);

        Map<VariableReference, DoubleTensor> position = ImmutableMap.of(v.getReference(), v.getValue());

        AdaptiveQuadraticPotential potential = new AdaptiveQuadraticPotential(
            zeros(position),
            ones(position),
            10,
            10,
            101,
            KeanuRandom.getDefaultRandom()
        );

        potential.update(position);

        Map<VariableReference, DoubleTensor> momentum = ImmutableMap.of(v.getReference(), DoubleTensor.scalar(0.5));


    }
}
