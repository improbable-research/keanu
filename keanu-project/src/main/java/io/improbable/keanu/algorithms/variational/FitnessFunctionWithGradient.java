package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradient;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static io.improbable.keanu.algorithms.variational.FitnessFunction.logOfTotalProbability;

public class FitnessFunctionWithGradient {

    protected final List<Vertex> probabilisticVertices;
    protected final List<? extends Vertex> latentVertices;
    protected final Map<String, Long> exploreSettingAll;

    public FitnessFunctionWithGradient(List<Vertex> probabilisticVertices,
                                       List<? extends Vertex> latentVertices) {
        this.probabilisticVertices = probabilisticVertices;
        this.latentVertices = latentVertices;
        this.exploreSettingAll = VertexValuePropagation.exploreSetting(latentVertices);
    }

    public MultivariateVectorFunction gradient() {
        return point -> {

            FitnessFunction.setAndCascadePoint(point, latentVertices, exploreSettingAll);

            Map<String, DoubleTensor> diffs = LogProbGradient.getJointLogProbGradientWrtLatents(probabilisticVertices);

            return alignGradientsToAppropriateIndex(diffs);
        };
    }

    public MultivariateFunction fitness() {
        return point -> {
            FitnessFunction.setAndCascadePoint(point, latentVertices, exploreSettingAll);
            return logOfTotalProbability(probabilisticVertices);
        };
    }

    private double[] alignGradientsToAppropriateIndex(Map<String /*Vertex Label*/, DoubleTensor /*Gradient*/> diffs) {

        List<DoubleTensor> tensors = new ArrayList<>();
        for (Vertex vertex : latentVertices) {
            DoubleTensor tensor = diffs.get(vertex.getId());
            if (tensor != null) {
                tensors.add(tensor);
            }else{
                int[] shape;
                if(vertex.getValue() instanceof DoubleTensor) {
                    shape  = ((DoubleTensor)vertex.getValue()).getShape();
                }else{
                    shape = new int[]{0};
                }
                tensors.add(DoubleTensor.zeros(shape));
            }
        }

        return flattenAll(tensors);
    }

    private double[] flattenAll(List<DoubleTensor> tensors) {
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