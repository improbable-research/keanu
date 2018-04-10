package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.vertices.Vertex;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FitnessFunctionWithGradient extends FitnessFunction {

    public FitnessFunctionWithGradient(List<Vertex<?>> fitnessVertices, List<? extends Vertex<Double>> latentVertices) {
        super(fitnessVertices, latentVertices);
    }

    public MultivariateVectorFunction gradient() {
        return point -> {

            setPoint(point);

            Map<String, Double> diffs = getDiffsWithRespectToUpstreamLatents();

            return alignGradientsToAppropriateIndex(diffs);
        };
    }

    private Map<String, Double> getDiffsWithRespectToUpstreamLatents() {
        Map<String, Double> diffOfLogWrt = new HashMap<>();

        for (final Vertex<?> v : probabilisticVertices) {

            //dlnDensityForV is the differential of the natural log of the fitness vertex's
            //density w.r.t an input vertex. The key of the map is the input vertex's
            //id.
            Map<String, Double> dlnDensityForV = v.dlnDensityAtValue();

            dlnDensityForV.forEach((vInput, dlogPdvInContrib) -> {
                //dlogPdvInContrib is the contribution to the rate of change of
                //the natural log of the fitness vertex due to vInput.
                double dlogPdvIn = diffOfLogWrt.getOrDefault(vInput, 0.0);
                diffOfLogWrt.put(vInput, dlogPdvIn + dlogPdvInContrib);
            });
        }

        return diffOfLogWrt;
    }

    private double[] alignGradientsToAppropriateIndex(Map<String /*Vertex Label*/, Double /*Gradient*/> diffs) {
        double[] gradient = new double[latentVertices.size()];
        for (int i = 0; i < gradient.length; i++) {
            gradient[i] = diffs.get(latentVertices.get(i).getId());
        }
        return gradient;
    }

}