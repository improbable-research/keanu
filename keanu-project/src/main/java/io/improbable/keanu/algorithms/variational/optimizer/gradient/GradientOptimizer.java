package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.FitnessFunction;
import io.improbable.keanu.backend.ProbabilisticGraph;
import io.improbable.keanu.backend.ProbabilisticWithGradientGraph;
import io.improbable.keanu.backend.keanu.KeanuProbabilisticWithGradientGraph;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.util.ProgressBar;
import io.improbable.keanu.vertices.Vertex;
import lombok.Builder;
import lombok.Getter;
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

import static org.apache.commons.math3.optim.nonlinear.scalar.GoalType.MAXIMIZE;

@Builder
public class GradientOptimizer implements Optimizer {

    private static final double FLAT_GRADIENT = 1e-16;

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
        List<Vertex> discreteLatentVertices = bayesNet.getDiscreteLatentVertices();
        boolean containsDiscreteLatents = !discreteLatentVertices.isEmpty();

        if (containsDiscreteLatents) {
            throw new UnsupportedOperationException("Gradient Optimization unsupported on Networks containing " +
                "Discrete Latents (" + discreteLatentVertices.size() + " found)");
        }

        bayesNet.cascadeObservations();

        return GradientOptimizer.builder()
            .probabilisticWithGradientGraph(new KeanuProbabilisticWithGradientGraph(bayesNet))
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

    @Getter
    private ProbabilisticWithGradientGraph probabilisticWithGradientGraph;

    /**
     * maxEvaluations the maximum number of objective function evaluations before throwing an exception
     * indicating convergence failure.
     */
    @Builder.Default
    private int maxEvaluations = Integer.MAX_VALUE;

    @Builder.Default
    private double relativeThreshold = 1e-8;

    @Builder.Default
    private double absoluteThreshold = 1e-8;

    /**
     * Specifies what formula to use to update the Beta parameter of the Nonlinear conjugate gradient method optimizer.
     */
    @Builder.Default
    private UpdateFormula updateFormula = UpdateFormula.POLAK_RIBIERE;

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

    @Override
    public ProbabilisticGraph getProbabilisticGraph() {
        return null;
    }

    private double optimize(FitnessFunctionWithGradient fitnessFunction) {

        ProgressBar progressBar = Optimizer.createFitnessProgressBar(this);

        ObjectiveFunction fitness = new ObjectiveFunction(fitnessFunction.fitness());
        ObjectiveFunctionGradient gradient = new ObjectiveFunctionGradient(fitnessFunction.gradient());

        double[] startingPoint = Optimizer.convertToPoint(
            probabilisticWithGradientGraph.getLatentVariables(),
            (Map<String, ? extends NumberTensor>) probabilisticWithGradientGraph.getLatentVariablesValues(),
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

    private static void warnIfGradientIsFlat(double[] gradient) {
        double maxGradient = Arrays.stream(gradient).max().orElseThrow(IllegalArgumentException::new);
        if (Math.abs(maxGradient) <= FLAT_GRADIENT) {
            throw new IllegalStateException("The initial gradient is very flat. The largest gradient is " + maxGradient);
        }
    }

}