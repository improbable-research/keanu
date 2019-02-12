package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static io.improbable.keanu.algorithms.variational.optimizer.Optimizer.convertFromPoint;

@AllArgsConstructor
public class ApacheFitnessFunctionGradientAdapter implements MultivariateVectorFunction {

    private final FitnessFunctionGradient fitnessFunctionGradient;
    private final List<? extends Variable> latentVariables;

    @Override
    public double[] value(double[] point) {

        Map<VariableReference, DoubleTensor> values = convertFromPoint(point, latentVariables);

        Map<? extends VariableReference, DoubleTensor> diffs = fitnessFunctionGradient.getGradientsAt(values);

        return alignGradientsToAppropriateIndex(diffs, latentVariables);
    }

    private static double[] alignGradientsToAppropriateIndex(Map<? extends VariableReference, DoubleTensor> diffs,
                                                             List<? extends Variable> latentVariables) {

        List<DoubleTensor> tensors = new ArrayList<>();
        for (Variable variable : latentVariables) {
            DoubleTensor tensor = diffs.get(variable.getReference());
            if (tensor != null) {
                tensors.add(tensor);
            } else {
                tensors.add(DoubleTensor.zeros(variable.getShape()));
            }
        }

        return flattenAll(tensors);
    }

    private static double[] flattenAll(List<DoubleTensor> tensors) {
        int totalLatentDimensions = 0;
        for (DoubleTensor tensor : tensors) {
            totalLatentDimensions += tensor.getLength();
        }

        double[] gradient = new double[totalLatentDimensions];
        int fillPointer = 0;
        for (DoubleTensor tensor : tensors) {
            double[] values = tensor.asFlatDoubleArray();
            System.arraycopy(values, 0, gradient, fillPointer, values.length);
            fillPointer += values.length;
        }

        return gradient;
    }
}
