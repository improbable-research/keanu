package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.variational.optimizer.*;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import lombok.AllArgsConstructor;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleValueChecker;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunctionGradient;
import org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static io.improbable.keanu.algorithms.variational.optimizer.Optimizer.getAsDoubleTensors;
import static org.apache.commons.math3.optim.nonlinear.scalar.GoalType.MAXIMIZE;

@AllArgsConstructor
public class ApacheNonLinearConjugateGradientOptimizer implements GradientOptimizationAlgorithm {

    private static final double FLAT_GRADIENT = 1e-16;

    public static ApacheNonLinearConjugateGradientOptimizerBuilder builder() {
        return new ApacheNonLinearConjugateGradientOptimizerBuilder();
    }

    public enum UpdateFormula {
        POLAK_RIBIERE(NonLinearConjugateGradientOptimizer.Formula.POLAK_RIBIERE),
        FLETCHER_REEVES(NonLinearConjugateGradientOptimizer.Formula.FLETCHER_REEVES);

        NonLinearConjugateGradientOptimizer.Formula apacheMapping;

        UpdateFormula(NonLinearConjugateGradientOptimizer.Formula apacheMapping) {
            this.apacheMapping = apacheMapping;
        }
    }

    private final int maxEvaluations;

    private final double relativeThreshold;

    private final double absoluteThreshold;

    /**
     * Specifies what formula to use to update the Beta parameter of the Nonlinear conjugate gradient method optimizer.
     */
    private UpdateFormula updateFormula;

    @Override
    public OptimizedResult optimize(final List<? extends Variable> latentVariables,
                                    FitnessFunction fitnessFunction,
                                    FitnessFunctionGradient fitnessFunctionGradient) {

        ObjectiveFunction fitness = new ObjectiveFunction(
            new ApacheFitnessFunctionAdaptor(fitnessFunction, latentVariables)
        );

        ObjectiveFunctionGradient gradient = new ObjectiveFunctionGradient(
            new ApacheFitnessFunctionGradientAdaptor(fitnessFunctionGradient, latentVariables)
        );

        double[] startingPoint = Optimizer.convertToPoint(getAsDoubleTensors(latentVariables));

        double initialFitness = fitness.getObjectiveFunction().value(startingPoint);
        double[] initialGradient = gradient.getObjectiveFunctionGradient().value(startingPoint);

        if (ProbabilityCalculator.isImpossibleLogProb(initialFitness)) {
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

        Map<VariableReference, DoubleTensor> optimizedValues = Optimizer
            .convertFromPoint(pointValuePair.getPoint(), latentVariables);

        return new OptimizedResult(optimizedValues, pointValuePair.getValue());
    }

    private static void warnIfGradientIsFlat(double[] gradient) {
        double maxGradient = Arrays.stream(gradient).max().orElseThrow(IllegalArgumentException::new);
        if (Math.abs(maxGradient) <= FLAT_GRADIENT) {
            throw new IllegalStateException("The initial gradient is very flat. The largest gradient is " + maxGradient);
        }
    }

    public static class ApacheNonLinearConjugateGradientOptimizerBuilder {

        private int maxEvaluations = Integer.MAX_VALUE;
        private double relativeThreshold = 1e-8;
        private double absoluteThreshold = 1e-8;
        private UpdateFormula updateFormula = UpdateFormula.POLAK_RIBIERE;

        ApacheNonLinearConjugateGradientOptimizerBuilder() {
        }

        public ApacheNonLinearConjugateGradientOptimizerBuilder maxEvaluations(int maxEvaluations) {
            this.maxEvaluations = maxEvaluations;
            return this;
        }

        public ApacheNonLinearConjugateGradientOptimizerBuilder relativeThreshold(double relativeThreshold) {
            this.relativeThreshold = relativeThreshold;
            return this;
        }

        public ApacheNonLinearConjugateGradientOptimizerBuilder absoluteThreshold(double absoluteThreshold) {
            this.absoluteThreshold = absoluteThreshold;
            return this;
        }

        public ApacheNonLinearConjugateGradientOptimizerBuilder updateFormula(UpdateFormula updateFormula) {
            this.updateFormula = updateFormula;
            return this;
        }

        public ApacheNonLinearConjugateGradientOptimizer build() {
            return new ApacheNonLinearConjugateGradientOptimizer(maxEvaluations, relativeThreshold, absoluteThreshold, updateFormula);
        }

        public String toString() {
            return "ApacheNonLinearConjugateGradientOptimizer.ApacheNonLinearConjugateGradientOptimizerBuilder(maxEvaluations=" + this.maxEvaluations + ", relativeThreshold=" + this.relativeThreshold + ", absoluteThreshold=" + this.absoluteThreshold + ", updateFormula=" + this.updateFormula + ")";
        }
    }
}
