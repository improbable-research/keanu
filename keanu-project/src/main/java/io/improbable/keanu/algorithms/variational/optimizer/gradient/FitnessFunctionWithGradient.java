package io.improbable.keanu.algorithms.variational.optimizer.gradient;


import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticWithGradientGraph;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

import static io.improbable.keanu.algorithms.variational.optimizer.Optimizer.convertFromPoint;


public class FitnessFunctionWithGradient {

    private final ProbabilisticWithGradientGraph probabilisticWithGradientGraph;
    private final boolean useLikelihood;

    private final BiConsumer<double[], double[]> onGradientCalculation;
    private final BiConsumer<double[], Double> onFitnessCalculation;

    public FitnessFunctionWithGradient(ProbabilisticWithGradientGraph probabilisticWithGradientGraph,
                                       boolean useLikelihood,
                                       BiConsumer<double[], double[]> onGradientCalculation,
                                       BiConsumer<double[], Double> onFitnessCalculation) {
        this.probabilisticWithGradientGraph = probabilisticWithGradientGraph;
        this.useLikelihood = useLikelihood;
        this.onGradientCalculation = onGradientCalculation;
        this.onFitnessCalculation = onFitnessCalculation;
    }

    public FitnessFunctionWithGradient(ProbabilisticWithGradientGraph probabilisticWithGradientGraph, boolean useLikelihood) {
        this(probabilisticWithGradientGraph, useLikelihood, null, null);
    }

    public MultivariateVectorFunction gradient() {
        return point -> {

            Map<VariableReference, DoubleTensor> values = getValues(point);

            Map<VariableReference, DoubleTensor> diffs = useLikelihood ?
                probabilisticWithGradientGraph.logLikelihoodGradients(values) :
                probabilisticWithGradientGraph.logProbGradients(values);

            double[] gradients = alignGradientsToAppropriateIndex(
                diffs,
                probabilisticWithGradientGraph.getLatentVariables(),
                probabilisticWithGradientGraph.getLatentVariablesShapes()
            );

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
                probabilisticWithGradientGraph.logLikelihood(values) :
                probabilisticWithGradientGraph.logProb(values);

            if (onFitnessCalculation != null) {
                onFitnessCalculation.accept(point, logOfTotalProbability);
            }

            return logOfTotalProbability;
        };
    }

    private Map<VariableReference, DoubleTensor> getValues(double[] point) {
        return convertFromPoint(
            point,
            probabilisticWithGradientGraph.getLatentVariables(),
            probabilisticWithGradientGraph.getLatentVariablesShapes()
        );
    }

    private static double[] alignGradientsToAppropriateIndex(Map<VariableReference, DoubleTensor> diffs,
                                                             List<VariableReference> latentVertices,
                                                             Map<VariableReference, long[]> latentShapes) {

        List<DoubleTensor> tensors = new ArrayList<>();
        for (VariableReference vertex : latentVertices) {
            DoubleTensor tensor = diffs.get(vertex);
            if (tensor != null) {
                tensors.add(tensor);
            } else {
                tensors.add(DoubleTensor.zeros(latentShapes.get(vertex)));
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