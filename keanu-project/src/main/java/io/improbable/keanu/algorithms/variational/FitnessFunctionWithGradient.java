package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.vertices.ContinuousVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradient;
import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;

import java.util.List;
import java.util.Map;

import static io.improbable.keanu.algorithms.variational.FitnessFunction.logOfTotalProbability;

public class FitnessFunctionWithGradient {

    protected final List<? extends ContinuousVertex<Double>> probabilisticVertices;
    protected final List<? extends Vertex<Double>> latentVertices;
    protected final Map<String, Long> exploreSettingAll;

    public FitnessFunctionWithGradient(List<? extends ContinuousVertex<Double>> probabilisticVertices,
                                       List<? extends Vertex<Double>> latentVertices) {
        this.probabilisticVertices = probabilisticVertices;
        this.latentVertices = latentVertices;
        this.exploreSettingAll = VertexValuePropagation.exploreSetting(latentVertices);
    }

    public MultivariateVectorFunction gradient() {
        return point -> {

            setAndCascadePoint(point);

            Map<String, Double> diffs = LogProbGradient.getJointLogProbGradientWrtLatents(probabilisticVertices);

            return alignGradientsToAppropriateIndex(diffs);
        };
    }

    public MultivariateFunction fitness() {
        return point -> {
            setAndCascadePoint(point);
            return logOfTotalProbability(probabilisticVertices);
        };
    }

    protected void setAndCascadePoint(double[] point) {
        for (int i = 0; i < point.length; i++) {
            Vertex<Double> vertex = latentVertices.get(i);
            vertex.setValue(point[i]);
        }

        VertexValuePropagation.cascadeUpdate(latentVertices, exploreSettingAll);
    }

    private double[] alignGradientsToAppropriateIndex(Map<String /*Vertex Label*/, Double /*Gradient*/> diffs) {
        double[] gradient = new double[latentVertices.size()];
        for (int i = 0; i < gradient.length; i++) {
            gradient[i] = diffs.get(latentVertices.get(i).getId());
        }
        return gradient;
    }

}