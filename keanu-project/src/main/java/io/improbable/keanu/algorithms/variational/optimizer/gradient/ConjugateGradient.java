package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunction;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import lombok.ToString;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleValueChecker;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunctionGradient;
import org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer;

import java.util.List;
import java.util.Map;

import static io.improbable.keanu.algorithms.variational.optimizer.Optimizer.getAsDoubleTensors;
import static org.apache.commons.math3.optim.nonlinear.scalar.GoalType.MAXIMIZE;

@AllArgsConstructor
/**
 * Backed by Apache Math org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer
 */
public class ConjugateGradient implements GradientOptimizationAlgorithm {

    public static ConjugateGradientBuilder builder() {
        return new ConjugateGradientBuilder();
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
            new ApacheFitnessFunctionAdapter(fitnessFunction, latentVariables)
        );

        ObjectiveFunctionGradient gradient = new ObjectiveFunctionGradient(
            new ApacheFitnessFunctionGradientAdapter(fitnessFunctionGradient, latentVariables)
        );

        double[] startingPoint = Optimizer.convertToArrayPoint(getAsDoubleTensors(latentVariables));

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

    @ToString
    public static class ConjugateGradientBuilder {

        private int maxEvaluations = Integer.MAX_VALUE;
        private double relativeThreshold = 1e-8;
        private double absoluteThreshold = 1e-8;
        private UpdateFormula updateFormula = UpdateFormula.POLAK_RIBIERE;

        ConjugateGradientBuilder() {
        }

        public ConjugateGradientBuilder maxEvaluations(int maxEvaluations) {
            this.maxEvaluations = maxEvaluations;
            return this;
        }

        public ConjugateGradientBuilder relativeThreshold(double relativeThreshold) {
            this.relativeThreshold = relativeThreshold;
            return this;
        }

        public ConjugateGradientBuilder absoluteThreshold(double absoluteThreshold) {
            this.absoluteThreshold = absoluteThreshold;
            return this;
        }

        public ConjugateGradientBuilder updateFormula(UpdateFormula updateFormula) {
            this.updateFormula = updateFormula;
            return this;
        }

        public ConjugateGradient build() {
            return new ConjugateGradient(maxEvaluations, relativeThreshold, absoluteThreshold, updateFormula);
        }

    }
}
