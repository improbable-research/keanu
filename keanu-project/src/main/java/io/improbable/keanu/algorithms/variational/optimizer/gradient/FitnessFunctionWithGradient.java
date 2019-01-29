package io.improbable.keanu.algorithms.variational.optimizer.gradient;


import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

import static io.improbable.keanu.algorithms.variational.optimizer.Optimizer.convertFromPoint;


public class FitnessFunctionWithGradient {

    private final ProbabilisticModelWithGradient probabilisticModel;
    private final boolean useLikelihood;

    private final BiConsumer<double[], double[]> onGradientCalculation;
    private final BiConsumer<double[], Double> onFitnessCalculation;

    public FitnessFunctionWithGradient(ProbabilisticModelWithGradient probabilisticModel,
                                       boolean useLikelihood,
                                       BiConsumer<double[], double[]> onGradientCalculation,
                                       BiConsumer<double[], Double> onFitnessCalculation) {
        this.probabilisticModel = probabilisticModel;
        this.useLikelihood = useLikelihood;
        this.onGradientCalculation = onGradientCalculation;
        this.onFitnessCalculation = onFitnessCalculation;
    }

    public FitnessFunctionWithGradient(ProbabilisticModelWithGradient probabilisticModel, boolean useLikelihood) {
        this(probabilisticModel, useLikelihood, null, null);
    }

    public MultivariateVectorFunction gradient() {
        return point -> {

            Map<VariableReference, DoubleTensor> values = getValues(point);

            Map<? extends VariableReference, DoubleTensor> diffs = useLikelihood ?
                probabilisticModel.logLikelihoodGradients(values) :
                probabilisticModel.logProbGradients(values);

            double[] gradients = alignGradientsToAppropriateIndex(diffs, probabilisticModel.getLatentVariables());

            if (onGradientCalculation != null) {
                onGradientCalculation.accept(point, gradients);
            }

            return gradients;
        };
    }

    public MultivariateFunction fitness() {
        return point -> {

            Map<VariableReference, DoubleTensor> values = getValues(point);

            double logOfTotalProbability = useLikelihood ?
                probabilisticModel.logLikelihood(values) :
                probabilisticModel.logProb(values);

            if (onFitnessCalculation != null) {
                onFitnessCalculation.accept(point, logOfTotalProbability);
            }

            return logOfTotalProbability;
        };
    }

    private Map<VariableReference, DoubleTensor> getValues(double[] point) {
        return convertFromPoint(point, probabilisticModel.getLatentVariables());
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