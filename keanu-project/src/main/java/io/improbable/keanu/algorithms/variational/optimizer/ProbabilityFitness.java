package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.algorithms.ProbabilisticModel;
import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.LogLikelihoodFitnessFunctionGradient;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.LogProbFitnessFunctionGradient;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.LogLikelihoodFitnessFunction;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.LogProbFitnessFunction;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;
import java.util.function.BiConsumer;

public enum ProbabilityFitness {

    MLE {
        @Override
        public FitnessFunction getFitnessFunction(ProbabilisticModel model,
                                                  BiConsumer<Map<VariableReference, DoubleTensor>, Double> handleFitnessCalculation) {
            return new LogLikelihoodFitnessFunction(model, handleFitnessCalculation);
        }

        @Override
        public FitnessFunctionGradient getFitnessFunctionGradient(ProbabilisticModelWithGradient model,
                                                                  BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>> handleFitnessCalculation) {
            return new LogLikelihoodFitnessFunctionGradient(model, handleFitnessCalculation);

        }
    },
    MAP {
        @Override
        public FitnessFunction getFitnessFunction(ProbabilisticModel model,
                                                  BiConsumer<Map<VariableReference, DoubleTensor>, Double> handleFitnessCalculation) {
            return new LogProbFitnessFunction(model, handleFitnessCalculation);
        }

        @Override
        public FitnessFunctionGradient getFitnessFunctionGradient(ProbabilisticModelWithGradient model,
                                                                  BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>> handleFitnessCalculation) {
            return new LogProbFitnessFunctionGradient(model, handleFitnessCalculation);
        }
    };

    public abstract FitnessFunction getFitnessFunction(ProbabilisticModel model,
                                                       BiConsumer<Map<VariableReference, DoubleTensor>, Double> handleFitnessCalculation);

    public FitnessFunction getFitnessFunction(ProbabilisticModel model) {
        return getFitnessFunction(model, (point, fitness) -> {
        });
    }

    public abstract FitnessFunctionGradient getFitnessFunctionGradient(ProbabilisticModelWithGradient model,
                                                                       BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>> handleFitnessCalculation);

    public FitnessFunctionGradient getFitnessFunctionGradient(ProbabilisticModelWithGradient model) {
        return getFitnessFunctionGradient(model, (point, gradient) -> {
        });
    }

}
