package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.variational.optimizer.*;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.ProgressBar;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;

@AllArgsConstructor(access = AccessLevel.PRIVATE)
public class GradientOptimizer implements Optimizer {

    private static final double FLAT_GRADIENT = 1e-16;

    public static GradientOptimizerBuilder builder() {
        return new GradientOptimizerBuilder();
    }

    private final ProbabilisticWithGradientGraph probabilisticWithGradientGraph;

    private final GradientOptimizationAlgorithm gradientOptimizationAlgorithm;

    private final boolean checkInitialFitnessConditions;

    private final List<BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>>> onGradientCalculations = new ArrayList<>();
    private final List<BiConsumer<Map<VariableReference, DoubleTensor>, Double>> onFitnessCalculations = new ArrayList<>();

    /**
     * Adds a callback to be called whenever the optimizer evaluates the gradient at a point.
     *
     * @param gradientCalculationHandler a function to be called whenever the optimizer evaluates the gradient at a point.
     *                                   The double[] argument to the handler represents the point being evaluated.
     *                                   The double[] argument to the handler represents the gradient of that point.
     */
    public void addGradientCalculationHandler(BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>> gradientCalculationHandler) {
        this.onGradientCalculations.add(gradientCalculationHandler);
    }

    /**
     * Removes a callback function that previously would have been called whenever the optimizer
     * evaluated the gradient at a point. If the callback is not registered then this function will do nothing.
     *
     * @param gradientCalculationHandler the function to be removed from the list of gradient evaluation callbacks
     */
    public void removeGradientCalculationHandler(BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>> gradientCalculationHandler) {
        this.onGradientCalculations.remove(gradientCalculationHandler);
    }

    private void handleGradientCalculation(Map<VariableReference, DoubleTensor> point, Map<? extends VariableReference, DoubleTensor> gradients) {
        for (BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>> gradientCalculationHandler : onGradientCalculations) {
            gradientCalculationHandler.accept(point, gradients);
        }
    }

    @Override
    public void addFitnessCalculationHandler(BiConsumer<Map<VariableReference, DoubleTensor>, Double> fitnessCalculationHandler) {
        this.onFitnessCalculations.add(fitnessCalculationHandler);
    }

    @Override
    public void removeFitnessCalculationHandler(BiConsumer<Map<VariableReference, DoubleTensor>, Double> fitnessCalculationHandler) {
        this.onFitnessCalculations.remove(fitnessCalculationHandler);
    }

    private void handleFitnessCalculation(Map<VariableReference, DoubleTensor> point, Double fitness) {
        for (BiConsumer<Map<VariableReference, DoubleTensor>, Double> fitnessCalculationHandler : onFitnessCalculations) {
            fitnessCalculationHandler.accept(point, fitness);
        }
    }

    private void assertHasLatents() {
        if (probabilisticWithGradientGraph.getLatentVariables().isEmpty()) {
            throw new IllegalArgumentException("Cannot find MAP of network without any latent variables");
        }
    }

    @Override
    public OptimizedResult maxAPosteriori() {
        return optimize(probabilisticWithGradientGraph, false);
    }

    @Override
    public OptimizedResult maxLikelihood() {
        return optimize(probabilisticWithGradientGraph, true);
    }

    private OptimizedResult optimize(ProbabilisticWithGradientGraph probabilisticWithGradientGraph, boolean useMLE) {
        assertHasLatents();

        FitnessFunction fitnessFunction = new FitnessFunction(
            probabilisticWithGradientGraph,
            useMLE,
            this::handleFitnessCalculation
        );

        FitnessFunctionGradient fitnessFunctionGradient = new FitnessFunctionGradient(
            probabilisticWithGradientGraph,
            useMLE,
            this::handleGradientCalculation
        );

        return optimize(fitnessFunction, fitnessFunctionGradient);
    }

    private OptimizedResult optimize(FitnessFunction fitnessFunction, FitnessFunctionGradient fitnessFunctionGradient) {

        ProgressBar progressBar = Optimizer.createFitnessProgressBar(this);

        if (checkInitialFitnessConditions) {
            Map<VariableReference, DoubleTensor> startingPoint = getPoint(probabilisticWithGradientGraph.getLatentVariables());

            double initialFitness = fitnessFunction.value(startingPoint);

            if (ProbabilityCalculator.isImpossibleLogProb(initialFitness)) {
                throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
            }

            Map<? extends VariableReference, DoubleTensor> initialGradient = fitnessFunctionGradient.value(startingPoint);
            warnIfGradientIsFlat(initialGradient);
        }

        OptimizedResult result = gradientOptimizationAlgorithm.optimize(
            probabilisticWithGradientGraph.getLatentVariables(),
            fitnessFunction,
            fitnessFunctionGradient
        );

        progressBar.finish();

        return result;
    }

    private Map<VariableReference, DoubleTensor> getPoint(List<? extends Variable> variables) {
        return variables.stream()
            .collect(Collectors.toMap(Variable::getReference, v -> {
                if (v.getValue() instanceof DoubleTensor) {
                    return (DoubleTensor) v.getValue();
                } else {
                    throw new UnsupportedOperationException(
                        "Optimization unsupported on networks containing discrete latents. " +
                            "Discrete latent : " + v.getReference() + " found.");
                }
            }));
    }

    private static void warnIfGradientIsFlat(Map<? extends VariableReference, DoubleTensor> gradient) {
        double maxGradient = gradient.values().stream()
            .flatMap(v -> Arrays.stream(v.asFlatDoubleArray()).boxed())
            .mapToDouble(v -> v).max().orElseThrow(IllegalArgumentException::new);

        if (Math.abs(maxGradient) <= FLAT_GRADIENT) {
            throw new IllegalStateException("The initial gradient is very flat. The largest gradient is " + maxGradient);
        }
    }

    public static class GradientOptimizerBuilder {

        private ProbabilisticWithGradientGraph probabilisticWithGradientGraph;
        private GradientOptimizationAlgorithm gradientOptimizationAlgorithm = NonLinearConjugateGradientOptimizer.builder().build();
        private boolean checkInitialFitnessConditions = true;

        GradientOptimizerBuilder() {
        }

        public GradientOptimizerBuilder bayesianNetwork(ProbabilisticWithGradientGraph probabilisticWithGradientGraph) {
            this.probabilisticWithGradientGraph = probabilisticWithGradientGraph;
            return this;
        }

        public GradientOptimizerBuilder algorithm(GradientOptimizationAlgorithm gradientOptimizationAlgorithm) {
            this.gradientOptimizationAlgorithm = gradientOptimizationAlgorithm;
            return this;
        }

        public GradientOptimizerBuilder checkInitialFitnessConditions(boolean check) {
            this.checkInitialFitnessConditions = check;
            return this;
        }

        public GradientOptimizer build() {
            if (probabilisticWithGradientGraph == null) {
                throw new IllegalStateException("Cannot build optimizer without specifying network to optimize.");
            }
            if (gradientOptimizationAlgorithm == null) {
                throw new IllegalStateException("Cannot build optimizer without specifying algorithm for optimizing.");
            }
            return new GradientOptimizer(
                probabilisticWithGradientGraph,
                gradientOptimizationAlgorithm,
                checkInitialFitnessConditions
            );
        }

    }
}