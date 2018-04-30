package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradient;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;

import java.util.List;
import java.util.Map;

public class FitnessFunctionWithGradient extends FitnessFunction {

    public FitnessFunctionWithGradient(List<Vertex<?>> fitnessVertices, List<? extends Vertex<Double>> latentVertices) {
        super(fitnessVertices, latentVertices);
    }

    public MultivariateVectorFunction gradient() {
        return point -> {

            setAndCascadePoint(point);

            Map<String, Double> diffs = LogProbGradient.getJointLogProbGradientWrtLatents(probabilisticVertices);

            return alignGradientsToAppropriateIndex(diffs);
        };
    }

    private double[] alignGradientsToAppropriateIndex(Map<String /*Vertex Label*/, Double /*Gradient*/> diffs) {
        double[] gradient = new double[latentVertices.size()];
        for (int i = 0; i < gradient.length; i++) {
            gradient[i] = diffs.get(latentVertices.get(i).getId());
        }
        return gradient;
    }

}