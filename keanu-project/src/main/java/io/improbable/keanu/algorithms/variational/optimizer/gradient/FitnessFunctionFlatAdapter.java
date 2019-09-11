package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunction;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;

import java.util.List;
import java.util.Map;

import static io.improbable.keanu.algorithms.variational.optimizer.Optimizer.convertFromPoint;

@AllArgsConstructor
public class FitnessFunctionFlatAdapter {

    private final FitnessFunction fitnessFunction;
    protected final List<? extends Variable> latentVariables;

    public double fitness(double[] point) {

        Map<VariableReference, DoubleTensor> values = convertFromPoint(point, latentVariables);

        return fitnessFunction.getFitnessAt(values);
    }

}
