package io.improbable.keanu.algorithms.variational;

import static io.improbable.keanu.algorithms.variational.FitnessFunction.logOfTotalProbability;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradient;


public class FitnessFunctionWithGradient {

    private final List<Probabilistic<?>> probabilisticVertices;
    private final List<? extends Vertex<DoubleTensor>> latentVertices;
    private final BiConsumer<double[], double[]> onGradientCalculation;
    private final BiConsumer<double[], Double> onFitnessCalculation;

    public FitnessFunctionWithGradient(List<Probabilistic<?>> probabilisticVertices,
                                       List<? extends Vertex<DoubleTensor>> latentVertices,
                                       BiConsumer<double[], double[]> onGradientCalculation,
                                       BiConsumer<double[], Double> onFitnessCalculation) {
        this.probabilisticVertices = probabilisticVertices;
        this.latentVertices = latentVertices;
        this.onGradientCalculation = onGradientCalculation;
        this.onFitnessCalculation = onFitnessCalculation;
    }

    public FitnessFunctionWithGradient(List<Probabilistic<?>> probabilisticVertices,
                                       List<? extends Vertex<DoubleTensor>> latentVertices) {
        this(probabilisticVertices, latentVertices, null, null);
    }

    public MultivariateVectorFunction gradient() {
        return point -> {

            FitnessFunction.setAndCascadePoint(point, latentVertices);

            Map<Long, DoubleTensor> diffs = LogProbGradient.getJointLogProbGradientWrtLatents(probabilisticVertices);

            double[] gradients = alignGradientsToAppropriateIndex(diffs, latentVertices);

            if (onGradientCalculation != null) {
                onGradientCalculation.accept(point, gradients);
            }

            return gradients;
        };
    }

    public MultivariateFunction fitness() {
        return point -> {
            FitnessFunction.setAndCascadePoint(point, latentVertices);
            double logOfTotalProbability = logOfTotalProbability(probabilisticVertices);

            if (onFitnessCalculation != null) {
                onFitnessCalculation.accept(point, logOfTotalProbability);
            }

            return logOfTotalProbability;
        };
    }

    private static double[] alignGradientsToAppropriateIndex(Map<Long /*Vertex Label*/, DoubleTensor /*Gradient*/> diffs,
                                                             List<? extends Vertex<DoubleTensor>> latentVertices) {

        List<DoubleTensor> tensors = new ArrayList<>();
        for (Vertex<DoubleTensor> vertex : latentVertices) {
            DoubleTensor tensor = diffs.get(vertex.getId());
            if (tensor != null) {
                tensors.add(tensor);
            } else {
                tensors.add(DoubleTensor.zeros(vertex.getShape()));
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