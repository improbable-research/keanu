package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunction;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import org.apache.commons.math3.analysis.MultivariateFunction;

import java.util.List;
import java.util.Map;

import static io.improbable.keanu.algorithms.variational.optimizer.Optimizer.convertFromPoint;

@AllArgsConstructor
public class ApacheFitnessFunctionAdapter implements MultivariateFunction {

    private final FitnessFunction fitnessFunction;
    private final List<? extends Variable> latentVariables;

    @Override
    public double value(double[] point) {

        Map<VariableReference, DoubleTensor> values = convertFromPoint(point, latentVariables);

        return fitnessFunction.getFitnessAt(values);
    }

}
