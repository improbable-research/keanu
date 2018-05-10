package io.improbable.keanu.algorithms.tensorvariational;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradient;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static io.improbable.keanu.algorithms.tensorvariational.TensorFitnessFunction.logOfTotalProbability;


public class TensorFitnessFunctionWithGradient {

    private final List<Vertex> probabilisticVertices;
    private final List<? extends Vertex<DoubleTensor>> latentVertices;
    private final Map<String, Long> exploreSettingAll;

    public TensorFitnessFunctionWithGradient(List<Vertex> probabilisticVertices,
                                             List<? extends Vertex<DoubleTensor>> latentVertices) {
        this.probabilisticVertices = probabilisticVertices;
        this.latentVertices = latentVertices;
        this.exploreSettingAll = VertexValuePropagation.exploreSetting(latentVertices);
    }

    public MultivariateVectorFunction gradient() {
        return point -> {

            TensorFitnessFunction.setAndCascadePoint(point, latentVertices, exploreSettingAll);

            Map<String, DoubleTensor> diffs = LogProbGradient.getJointLogProbGradientWrtLatents(probabilisticVertices);

            return alignGradientsToAppropriateIndex(diffs, latentVertices);
        };
    }

    public MultivariateFunction fitness() {
        return point -> {
            TensorFitnessFunction.setAndCascadePoint(point, latentVertices, exploreSettingAll);
            return logOfTotalProbability(probabilisticVertices);
        };
    }

    private static double[] alignGradientsToAppropriateIndex(Map<String /*Vertex Label*/, DoubleTensor /*Gradient*/> diffs,
                                                     List<? extends Vertex<DoubleTensor>> latentVertices) {

        List<DoubleTensor> tensors = new ArrayList<>();
        for (Vertex<DoubleTensor> vertex : latentVertices) {
            DoubleTensor tensor = diffs.get(vertex.getId());
            if (tensor != null) {
                tensors.add(tensor);
            } else {
                int[] missingVertexShape = vertex.getValue().getShape();
                tensors.add(DoubleTensor.zeros(missingVertexShape));
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
            double[] values = tensor.getLinearView();
            System.arraycopy(values, 0, gradient, fillPointer, values.length);
            fillPointer += values.length;
        }

        return gradient;
    }

}