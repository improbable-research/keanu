package io.improbable.keanu.algorithms.variational.optimizer.gradient;


import static io.improbable.keanu.algorithms.variational.optimizer.Optimizer.setAndCascadePoint;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;


public class FitnessFunctionWithGradient {

    private final List<? extends Vertex> ofVertices;
    private final List<? extends Vertex<DoubleTensor>> wrtVertices;
    private final LogProbGradientCalculator logProbGradient;

    private final BiConsumer<double[], double[]> onGradientCalculation;
    private final BiConsumer<double[], Double> onFitnessCalculation;

    public FitnessFunctionWithGradient(List<? extends Vertex> ofVertices,
                                       List<? extends Vertex<DoubleTensor>> wrtVertices,
                                       BiConsumer<double[], double[]> onGradientCalculation,
                                       BiConsumer<double[], Double> onFitnessCalculation) {
        this.ofVertices = ofVertices;
        this.wrtVertices = wrtVertices;
        this.logProbGradient = new LogProbGradientCalculator(ofVertices, wrtVertices);
        this.onGradientCalculation = onGradientCalculation;
        this.onFitnessCalculation = onFitnessCalculation;
    }

    public FitnessFunctionWithGradient(List<? extends Vertex> ofVertices,
                                       List<? extends Vertex<DoubleTensor>> wrtVertices) {
        this(ofVertices, wrtVertices, null, null);
    }

    public MultivariateVectorFunction gradient() {
        return point -> {

            setAndCascadePoint(point, wrtVertices);

            Map<VertexId, DoubleTensor> diffs = logProbGradient.getJointLogProbGradientWrtLatents();

            double[] gradients = alignGradientsToAppropriateIndex(diffs, wrtVertices);

            if (onGradientCalculation != null) {
                onGradientCalculation.accept(point, gradients);
            }

            return gradients;
        };
    }

    public MultivariateFunction fitness() {
        return point -> {
            setAndCascadePoint(point, wrtVertices);
            double logOfTotalProbability = ProbabilityCalculator.calculateLogProbFor(ofVertices);

            if (onFitnessCalculation != null) {
                onFitnessCalculation.accept(point, logOfTotalProbability);
            }

            return logOfTotalProbability;
        };
    }

    private static double[] alignGradientsToAppropriateIndex(Map<VertexId, DoubleTensor /*Gradient*/> diffs,
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