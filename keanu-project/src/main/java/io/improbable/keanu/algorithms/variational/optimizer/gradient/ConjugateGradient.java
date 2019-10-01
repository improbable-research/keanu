package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.VariableTransform;
import lombok.AllArgsConstructor;
import lombok.ToString;
import org.apache.commons.math3.exception.NotStrictlyPositiveException;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleValueChecker;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunctionGradient;
import org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

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
    public OptimizedResult optimize(List<? extends Variable<DoubleTensor, ?>> latentVariables,
                                    FitnessFunctionGradient fitnessFunctionGradient) {

        Map<VariableReference, VariableTransform> transforms = latentVariables.stream()
            .filter(v -> v instanceof UniformVertex)
            .collect(Collectors.toMap(Variable::getReference, v -> {

                DoubleTensor min = ((UniformVertex) v).getXMin().getValue();
                DoubleTensor max = ((UniformVertex) v).getXMax().getValue();
                DoubleTensor range = max.minus(min);

                DoubleVertex placeholder = new DoublePlaceholderVertex();
                DoubleVertex output = placeholder.sigmoid().times(ConstantVertex.of(range)).plus(ConstantVertex.of(min));

                VariableTransform transform = new VariableTransform(placeholder, output);

                return transform;
            }));

        ReparameterizationAdapter reparameterizationAdapter = new ReparameterizationAdapter(fitnessFunctionGradient, transforms);

        ObjectiveFunction fitness = new ObjectiveFunction(
            new ApacheFitnessFunctionAdaptor(new FitnessFunctionFlatAdapter(reparameterizationAdapter, latentVariables))
        );

        ObjectiveFunctionGradient gradient = new ObjectiveFunctionGradient(
            new ApacheFitnessFunctionGradientAdaptor(new FitnessFunctionGradientFlatAdapter(reparameterizationAdapter, latentVariables))
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

        Map<VariableReference, DoubleTensor> optimizedValues = Optimizer.convertFromPoint(
            pointValuePair.getPoint(),
            latentVariables
        );

        Map<VariableReference, DoubleTensor> transformedOptimizedValues = reparameterizationAdapter.transform(optimizedValues);

        return new OptimizedResult(transformedOptimizedValues, pointValuePair.getValue());
    }

    @ToString
    public static class ConjugateGradientBuilder {

        private int maxEvaluations = Integer.MAX_VALUE;
        private double relativeThreshold = 1e-8;
        private double absoluteThreshold = 1e-8;
        private UpdateFormula updateFormula = UpdateFormula.POLAK_RIBIERE;

        public ConjugateGradientBuilder maxEvaluations(int maxEvaluations) {
            if (maxEvaluations <= 0) {
                throw new NotStrictlyPositiveException(maxEvaluations);
            }
            this.maxEvaluations = maxEvaluations;
            return this;
        }

        public ConjugateGradientBuilder relativeThreshold(double relativeThreshold) {
            if (relativeThreshold <= 0) {
                throw new NotStrictlyPositiveException(relativeThreshold);
            }
            this.relativeThreshold = relativeThreshold;
            return this;
        }

        public ConjugateGradientBuilder absoluteThreshold(double absoluteThreshold) {
            if (absoluteThreshold <= 0) {
                throw new NotStrictlyPositiveException(absoluteThreshold);
            }
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
