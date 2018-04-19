package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.variational.FitnessFunctionWithGradient;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;

import java.util.*;


/**
 * Hamiltonian Monte Carlo is a method for obtaining samples from a probability
 * distribution with the introduction of a momentum variable.
 * <p>
 * Algorithm 1: "Hamiltonian Monte Carlo".
 * The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
 * https://arxiv.org/pdf/1111.4246.pdf
 */
public class Hamiltonian {

    private Hamiltonian() {
    }

    public static NetworkSamples getPosteriorSamples(final BayesNet bayesNet,
                                                     final List<DoubleVertex> fromVertices,
                                                     final int sampleCount,
                                                     final int leapFrogCount,
                                                     final double stepSize) {
        List<Double> startValues = new ArrayList<>();
        for (DoubleVertex vertex : fromVertices) {
            startValues.add(vertex.sample());
        }
        return getPosteriorSamples(bayesNet, fromVertices, sampleCount, leapFrogCount, stepSize, startValues, new Random());
    }

    public static NetworkSamples getPosteriorSamples(final BayesNet bayesNet,
                                                     final List<DoubleVertex> fromVertices,
                                                     final int sampleCount,
                                                     final int leapFrogCount,
                                                     final double stepSize,
                                                     final List<Double> startValues,
                                                     final Random random) {

        final Map<String, List<Double>> samples = setStartingSample(fromVertices, startValues);
        final Map<String, Integer> indexLookup = createIndexLookup(bayesNet.getContinuousLatentVertices());
        final MultivariateVectorFunction logOfMasterPGradientFunction = getLogOfMasterPGradientFunction(bayesNet);

        for (int sampleNum = 1; sampleNum < sampleCount; sampleNum++) {

            final Map<String, Double> position = getLatestSample(samples);
            final Map<String, Double> positionPreviousTimeStep = new HashMap<>(position);

            final Map<String, Double> momentum = initializeMomentumForEachVertex(fromVertices, random);
            final Map<String, Double> momentumPreviousTimeStep = new HashMap<>(momentum);

            final double logOfMasterP = bayesNet.getLogOfMasterP();

            for (int leapFrogNum = 0; leapFrogNum < leapFrogCount; leapFrogNum++) {
                leapfrogVertices(
                        fromVertices,
                        position,
                        momentum,
                        stepSize,
                        bayesNet,
                        logOfMasterPGradientFunction,
                        indexLookup
                );
            }

            final double likelihoodOfLeapfrog = getLikelihoodOfLeapfrog(
                    bayesNet,
                    momentum,
                    logOfMasterP,
                    momentumPreviousTimeStep
            );

            if (shouldAccept(likelihoodOfLeapfrog, random)) {
                addAsLatest(samples, position);
            } else {
                addAsLatest(samples, positionPreviousTimeStep);
            }
        }

        return new NetworkSamples(samples, sampleCount);
    }

    private static Map<String, List<Double>> setStartingSample(List<DoubleVertex> vertexes,
                                                               List<Double> startValues) {
        Map<String, List<Double>> samples = new HashMap<>();
        for (int i = 0; i < vertexes.size(); i++) {
            DoubleVertex currentVertex = vertexes.get(i);
            double startValue = startValues.get(i);

            samples.computeIfAbsent(currentVertex.getId(), id -> new ArrayList<>()).add(startValue);
        }
        return samples;
    }

    private static Map<String, Integer> createIndexLookup(List<Vertex<Double>> continuousLatentVertices) {
        Map<String, Integer> indexLookup = new HashMap<>();
        for (int i = 0; i < continuousLatentVertices.size(); i++) {
            indexLookup.put(continuousLatentVertices.get(i).getId(), i);
        }
        return indexLookup;
    }

    private static MultivariateVectorFunction getLogOfMasterPGradientFunction(BayesNet bayesNet) {
        final FitnessFunctionWithGradient fitnessFunction = new FitnessFunctionWithGradient(
                bayesNet.getVerticesThatContributeToMasterP(),
                bayesNet.getContinuousLatentVertices()
        );

        return fitnessFunction.gradient();
    }

    private static Map<String, Double> getLatestSample(Map<String, List<Double>> samples) {
        Map<String, Double> latestSample = new HashMap<>();
        for (String key : samples.keySet()) {
            latestSample.put(key, lastElementInList(samples.get(key)));
        }
        return latestSample;
    }

    private static Map<String, Double> initializeMomentumForEachVertex(List<DoubleVertex> vertexes,
                                                                       Random random) {
        Map<String, Double> momentums = new HashMap<>();
        for (int i = 0; i < vertexes.size(); i++) {
            Vertex currentVertex = vertexes.get(i);
            momentums.put(currentVertex.getId(), random.nextGaussian());
        }
        return momentums;
    }

    private static void leapfrogVertices(final List<DoubleVertex> fromVertices,
                                         final Map<String, Double> allPositions,
                                         final Map<String, Double> momentum,
                                         final double stepSize,
                                         final BayesNet bayesNet,
                                         final MultivariateVectorFunction logOfMasterPGradientFunction,
                                         final Map<String, Integer> indexLookup) {

        for (DoubleVertex currentVertex : fromVertices) {

            final double vertexPosition = allPositions.get(currentVertex.getId());
            final double vertexMomentum = momentum.get(currentVertex.getId());

            final Leapfrog leapfrog = leapfrogVertex(
                    currentVertex.getId(),
                    vertexPosition,
                    vertexMomentum,
                    stepSize,
                    bayesNet,
                    allPositions,
                    logOfMasterPGradientFunction,
                    indexLookup.get(currentVertex.getId())
            );

            allPositions.put(currentVertex.getId(), leapfrog.getPosition());
            momentum.put(currentVertex.getId(), leapfrog.getMomentum());

        }
    }

    private static Leapfrog leapfrogVertex(final String vertexId,
                                           final double vertexPosition,
                                           final double vertexMomentum,
                                           final double stepSize,
                                           final BayesNet bayesNet,
                                           final Map<String, Double> allPositions,
                                           final MultivariateVectorFunction logOfMasterPGradientFunction,
                                           final Integer indexLookup) {

        final double halfTimeStep = stepSize / 2.0;
        final double[] positions = convertPositionValuesToOrderedArray(bayesNet.getContinuousLatentVertices(), allPositions);

        final double[] gradients = logOfMasterPGradientFunction.value(positions);

        final double gradient = gradients[indexLookup];

        final double logOfMasterPBeforeLeap = bayesNet.getLogOfMasterP();

        final double momentumHalfTimeStep = vertexMomentum - (halfTimeStep * gradient * logOfMasterPBeforeLeap);

        final double positionTimeStep = vertexPosition + (stepSize * momentumHalfTimeStep);

        allPositions.put(vertexId, positionTimeStep);

        final double[] newPositions = convertPositionValuesToOrderedArray(bayesNet.getContinuousLatentVertices(), allPositions);

        final double[] newGradients = logOfMasterPGradientFunction.value(newPositions);

        final double newGradient = newGradients[indexLookup];

        final double logOfMasterPAfterLeap = bayesNet.getLogOfMasterP();

        final double momentumTimeStep = momentumHalfTimeStep - (halfTimeStep * newGradient * logOfMasterPAfterLeap);

        return new Leapfrog(positionTimeStep, momentumTimeStep);
    }

    private static double[] convertPositionValuesToOrderedArray(List<Vertex<Double>> continuousLatentVertices,
                                                                Map<String, Double> values) {
        double[] point = new double[continuousLatentVertices.size()];
        for (int i = 0; i < point.length; i++) {
            point[i] = values.get(continuousLatentVertices.get(i).getId());
        }
        return point;
    }

    private static double getLikelihoodOfLeapfrog(final BayesNet bayesNet,
                                                  final Map<String, Double> leapfroggedMomentum,
                                                  final double previousLogOfMasterP,
                                                  final Map<String, Double> momentumPreviousTimeStep) {
        final double logOfMasterP = bayesNet.getLogOfMasterP();

        final double leapFroggedMomentum = (0.5 * dotProduct(leapfroggedMomentum));
        final double previousMomentumDotProduct = (0.5 * dotProduct(momentumPreviousTimeStep));

        final double leapFroggedLikelihood = logOfMasterP - leapFroggedMomentum;
        final double previousLikelihood = previousLogOfMasterP - previousMomentumDotProduct;

        final double logLikelihoodOfLeapFrog = leapFroggedLikelihood - previousLikelihood;
        final double likelihoodOfLeapfrog = Math.exp(logLikelihoodOfLeapFrog);

        return Math.min(likelihoodOfLeapfrog, 1.0);
    }

    private static boolean shouldAccept(double likelihood, Random random) {
        return likelihood > random.nextDouble();
    }

    private static void addAsLatest(Map<String, List<Double>> samples,
                                    Map<String, Double> latestSample) {
        for (Map.Entry<String, List<Double>> entry : samples.entrySet()) {
            entry.getValue().add(latestSample.get(entry.getKey()));
        }
    }

    private static double dotProduct(Map<String, Double> momentums) {
        double dotProduct = 0.0;
        for (Double momentum : momentums.values()) {
            dotProduct += momentum * momentum;
        }
        return dotProduct;
    }

    private static <T> T lastElementInList(List<T> list) {
        return list.get(list.size() - 1);
    }

}
