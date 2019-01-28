package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import org.apache.commons.math3.analysis.MultivariateFunction;

import java.util.Map;

import static io.improbable.keanu.algorithms.variational.optimizer.Optimizer.convertFromPoint;

@AllArgsConstructor
public class ApacheFitnessFunctionAdaptor implements MultivariateFunction {

    private final FitnessFunction fitnessFunction;
    private final ProbabilisticGraph probabilisticGraph;

    @Override
    public double value(double[] point) {

        Map<VariableReference, DoubleTensor> values = getValues(point);

        return fitnessFunction.value(values);
    }

    private Map<VariableReference, DoubleTensor> getValues(double[] point) {
        return convertFromPoint(point, probabilisticGraph.getLatentVariables());
    }
}
