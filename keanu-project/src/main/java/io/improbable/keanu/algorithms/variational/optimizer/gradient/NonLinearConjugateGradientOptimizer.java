package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.variational.optimizer.*;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleValueChecker;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunctionGradient;

import java.util.List;
import java.util.Map;

import static io.improbable.keanu.algorithms.variational.optimizer.Optimizer.getAsDoubleTensors;
import static org.apache.commons.math3.optim.nonlinear.scalar.GoalType.MAXIMIZE;

@AllArgsConstructor
/**
 * Backed by Apache Math org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer
 */
public class NonLinearConjugateGradientOptimizer implements GradientOptimizationAlgorithm {

    public static ApacheNonLinearConjugateGradientOptimizerBuilder builder() {
        return new ApacheNonLinearConjugateGradientOptimizerBuilder();
    }

    public enum UpdateFormula {
        POLAK_RIBIERE(org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer.Formula.POLAK_RIBIERE),
        FLETCHER_REEVES(org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer.Formula.FLETCHER_REEVES);

        org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer.Formula apacheMapping;

        UpdateFormula(org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer.Formula apacheMapping) {
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

        org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer optimizer;

        optimizer = new org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer(
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

        public NonLinearConjugateGradientOptimizer build() {
            return new NonLinearConjugateGradientOptimizer(maxEvaluations, relativeThreshold, absoluteThreshold, updateFormula);
        }

        public String toString() {
            return "NonLinearConjugateGradientOptimizer.ApacheNonLinearConjugateGradientOptimizerBuilder(maxEvaluations=" + this.maxEvaluations + ", relativeThreshold=" + this.relativeThreshold + ", absoluteThreshold=" + this.absoluteThreshold + ", updateFormula=" + this.updateFormula + ")";
        }
    }
}
