package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.FitnessFunction;
import io.improbable.keanu.backend.ProbabilisticWithGradientGraph;
import io.improbable.keanu.backend.keanu.KeanuProbabilisticWithGradientGraph;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.util.ProgressBar;
import io.improbable.keanu.vertices.Vertex;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleValueChecker;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunctionGradient;
import org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

import static java.util.stream.Collectors.toMap;
import static org.apache.commons.math3.optim.nonlinear.scalar.GoalType.MAXIMIZE;

@AllArgsConstructor(access = AccessLevel.PRIVATE)
public class GradientOptimizer implements Optimizer {

    private static final double FLAT_GRADIENT = 1e-16;

    public static GradientOptimizerBuilder builder() {
        return new GradientOptimizerBuilder();
    }

    public enum UpdateFormula {
        POLAK_RIBIERE(NonLinearConjugateGradientOptimizer.Formula.POLAK_RIBIERE),
        FLETCHER_REEVES(NonLinearConjugateGradientOptimizer.Formula.FLETCHER_REEVES);

        NonLinearConjugateGradientOptimizer.Formula apacheMapping;

        UpdateFormula(NonLinearConjugateGradientOptimizer.Formula apacheMapping) {
            this.apacheMapping = apacheMapping;
        }
    }

    /**
     * Creates a {@link GradientOptimizer} which provides methods for optimizing the values of latent variables
     * of the Bayesian network to maximise probability.
     *
     * @param bayesNet The Bayesian network to run optimization on.
     * @return a {@link GradientOptimizer}
     */
    public static GradientOptimizer of(BayesianNetwork bayesNet) {
        return GradientOptimizer.builder()
            .bayesianNetwork(bayesNet)
            .build();
    }

    /**
     * Creates a Bayesian network from the given vertices and uses this to
     * create a {@link GradientOptimizer}. This provides methods for optimizing the values of latent variables
     * of the Bayesian network to maximise probability.
     *
     * @param vertices The vertices to create a Bayesian network from.
     * @return a {@link GradientOptimizer}
     */
    public static GradientOptimizer of(Collection<? extends Vertex> vertices) {
        return of(new BayesianNetwork(vertices));
    }

    /**
     * Creates a Bayesian network from the graph connected to the given vertex and uses this to
     * create a {@link GradientOptimizer}. This provides methods for optimizing the values of latent variables
     * of the Bayesian network to maximise probability.
     *
     * @param vertexFromNetwork A vertex in the graph to create the Bayesian network from
     * @return a {@link GradientOptimizer}
     */
    public static GradientOptimizer ofConnectedGraph(Vertex<?> vertexFromNetwork) {
        return of(vertexFromNetwork.getConnectedGraph());
    }

    private ProbabilisticWithGradientGraph probabilisticWithGradientGraph;

    /**
     * maxEvaluations the maximum number of objective function evaluations before throwing an exception
     * indicating convergence failure.
     */
    private int maxEvaluations;

    private double relativeThreshold;

    private double absoluteThreshold;

    /**
     * Specifies what formula to use to update the Beta parameter of the Nonlinear conjugate gradient method optimizer.
     */
    private UpdateFormula updateFormula;

    private final List<BiConsumer<double[], double[]>> onGradientCalculations = new ArrayList<>();
    private final List<BiConsumer<double[], Double>> onFitnessCalculations = new ArrayList<>();

    /**
     * Adds a callback to be called whenever the optimizer evaluates the gradient at a point.
     *
     * @param gradientCalculationHandler a function to be called whenever the optimizer evaluates the gradient at a point.
     *                                   The double[] argument to the handler represents the point being evaluated.
     *                                   The double[] argument to the handler represents the gradient of that point.
     */
    public void addGradientCalculationHandler(BiConsumer<double[], double[]> gradientCalculationHandler) {
        this.onGradientCalculations.add(gradientCalculationHandler);
    }

    /**
     * Removes a callback function that previously would have been called whenever the optimizer
     * evaluated the gradient at a point. If the callback is not registered then this function will do nothing.
     *
     * @param gradientCalculationHandler the function to be removed from the list of gradient evaluation callbacks
     */
    public void removeGradientCalculationHandler(BiConsumer<double[], double[]> gradientCalculationHandler) {
        this.onGradientCalculations.remove(gradientCalculationHandler);
    }

    private void handleGradientCalculation(double[] point, double[] gradients) {
        for (BiConsumer<double[], double[]> gradientCalculationHandler : onGradientCalculations) {
            gradientCalculationHandler.accept(point, gradients);
        }
    }

    @Override
    public void addFitnessCalculationHandler(BiConsumer<double[], Double> fitnessCalculationHandler) {
        this.onFitnessCalculations.add(fitnessCalculationHandler);
    }

    @Override
    public void removeFitnessCalculationHandler(BiConsumer<double[], Double> fitnessCalculationHandler) {
        this.onFitnessCalculations.remove(fitnessCalculationHandler);
    }

    private void handleFitnessCalculation(double[] point, Double fitness) {
        for (BiConsumer<double[], Double> fitnessCalculationHandler : onFitnessCalculations) {
            fitnessCalculationHandler.accept(point, fitness);
        }
    }

    private void assertHasLatents() {
        if (probabilisticWithGradientGraph.getLatentVariables().isEmpty()) {
            throw new IllegalArgumentException("Cannot find MAP of network without any latent variables");
        }
    }

    @Override
    public double maxAPosteriori() {
        assertHasLatents();

        FitnessFunctionWithGradient fitnessFunction = new FitnessFunctionWithGradient(
            probabilisticWithGradientGraph,
            false,
            this::handleGradientCalculation,
            this::handleFitnessCalculation
        );

        return optimize(fitnessFunction);
    }

    @Override
    public double maxLikelihood() {
        assertHasLatents();

        FitnessFunctionWithGradient fitnessFunction = new FitnessFunctionWithGradient(
            probabilisticWithGradientGraph,
            true,
            this::handleGradientCalculation,
            this::handleFitnessCalculation
        );

        return optimize(fitnessFunction);
    }

    private double optimize(FitnessFunctionWithGradient fitnessFunction) {

        ProgressBar progressBar = Optimizer.createFitnessProgressBar(this);

        ObjectiveFunction fitness = new ObjectiveFunction(fitnessFunction.fitness());
        ObjectiveFunctionGradient gradient = new ObjectiveFunctionGradient(fitnessFunction.gradient());

        double[] startingPoint = Optimizer.convertToPoint(
            probabilisticWithGradientGraph.getLatentVariables(),
            getLatentsAsNumberTensors(),
            probabilisticWithGradientGraph.getLatentVariablesShapes()
        );

        double initialFitness = fitness.getObjectiveFunction().value(startingPoint);
        double[] initialGradient = gradient.getObjectiveFunctionGradient().value(startingPoint);

        if (FitnessFunction.isValidInitialFitness(initialFitness)) {
            throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
        }

        warnIfGradientIsFlat(initialGradient);

        NonLinearConjugateGradientOptimizer optimizer;

        optimizer = new NonLinearConjugateGradientOptimizer(
            updateFormula.apacheMapping,
            new SimpleValueChecker(relativeThreshold, absoluteThreshold)
        );

        PointValuePair pointValuePair = optimizer.optimize(
            new MaxEval(maxEvaluations),
            fitness,
            gradient,
            MAXIMIZE,
            new InitialGuess(startingPoint)
        );

        progressBar.finish();
        return pointValuePair.getValue();
    }

    private Map<String, ? extends NumberTensor> getLatentsAsNumberTensors() {
        return probabilisticWithGradientGraph.getLatentVariablesValues().entrySet().stream()
            .collect(toMap(
                e -> e.getKey(),
                e -> {
                    Object value = e.getValue();
                    if (value instanceof NumberTensor) {
                        return (NumberTensor) e.getValue();
                    } else {
                        throw new UnsupportedOperationException(
                            "Gradient optimization unsupported on networks containing discrete latents. " +
                                "Discrete latent : " + e.getKey() + " found.");
                    }
                }
            ));
    }

    private static void warnIfGradientIsFlat(double[] gradient) {
        double maxGradient = Arrays.stream(gradient).max().orElseThrow(IllegalArgumentException::new);
        if (Math.abs(maxGradient) <= FLAT_GRADIENT) {
            throw new IllegalStateException("The initial gradient is very flat. The largest gradient is " + maxGradient);
        }
    }

    public static class GradientOptimizerBuilder {

        private ProbabilisticWithGradientGraph probabilisticWithGradientGraph;
        private int maxEvaluations = Integer.MAX_VALUE;
        private double relativeThreshold = 1e-8;
        private double absoluteThreshold = 1e-8;
        private UpdateFormula updateFormula = UpdateFormula.POLAK_RIBIERE;

        GradientOptimizerBuilder() {
        }

        public GradientOptimizerBuilder bayesianNetwork(BayesianNetwork network) {
            Optimizer.initializeNetworkForOptimization(network);
            return bayesianNetwork(new KeanuProbabilisticWithGradientGraph(network));
        }

        public GradientOptimizerBuilder bayesianNetwork(Collection<? extends Vertex> vertices) {
            return bayesianNetwork(new BayesianNetwork(vertices));
        }

        public GradientOptimizerBuilder bayesianNetwork(ProbabilisticWithGradientGraph probabilisticWithGradientGraph) {
            this.probabilisticWithGradientGraph = probabilisticWithGradientGraph;
            return this;
        }

        public GradientOptimizerBuilder maxEvaluations(int maxEvaluations) {
            this.maxEvaluations = maxEvaluations;
            return this;
        }

        public GradientOptimizerBuilder relativeThreshold(double relativeThreshold) {
            this.relativeThreshold = relativeThreshold;
            return this;
        }

        public GradientOptimizerBuilder absoluteThreshold(double absoluteThreshold) {
            this.absoluteThreshold = absoluteThreshold;
            return this;
        }

        public GradientOptimizerBuilder updateFormula(UpdateFormula updateFormula) {
            this.updateFormula = updateFormula;
            return this;
        }

        public GradientOptimizer build() {
            if (probabilisticWithGradientGraph == null) {
                throw new IllegalStateException("Cannot build optimizer without specifying network to optimize.");
            }
            return new GradientOptimizer(
                probabilisticWithGradientGraph,
                maxEvaluations,
                relativeThreshold,
                absoluteThreshold,
                updateFormula
            );
        }

        public String toString() {
            return "GradientOptimizer.GradientOptimizerBuilder(probabilisticWithGradientGraph=" + this.probabilisticWithGradientGraph + ", maxEvaluations=" + this.maxEvaluations + ", relativeThreshold=" + this.relativeThreshold + ", absoluteThreshold=" + this.absoluteThreshold + ", updateFormula=" + this.updateFormula + ")";
        }
    }
}