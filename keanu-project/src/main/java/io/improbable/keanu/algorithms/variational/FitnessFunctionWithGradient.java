package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.LogProbGradient;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;

import java.util.List;
import java.util.Map;

public class FitnessFunctionWithGradient extends FitnessFunction {

    public FitnessFunctionWithGradient(List<? extends Vertex> fitnessVertices, List<? extends Vertex<Double>> latentVertices) {
        super(fitnessVertices, latentVertices);
    }

    public MultivariateVectorFunction gradient() {
        return point -> {

            setAndCascadePoint(point);

            Map<Long, DoubleTensor> diffs = LogProbGradient.getJointLogProbGradientWrtLatents(probabilisticVertices);

            return alignGradientsToAppropriateIndex(DoubleTensor.toScalars(diffs));
        };
    }

    private double[] alignGradientsToAppropriateIndex(Map<Long /*Vertex Label*/, Double /*Gradient*/> diffs) {
        double[] gradient = new double[latentVertices.size()];
        for (int i = 0; i < gradient.length; i++) {
            gradient[i] = diffs.getOrDefault(latentVertices.get(i).getId(), 0.0);
        }
        return gradient;
    }

}