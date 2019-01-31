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
                                                  BiConsumer<Map<VariableReference, DoubleTensor>, Double> calculation) {
            return new LogLikelihoodFitnessFunction(model, calculation);
        }

        @Override
        public FitnessFunctionGradient getFitnessFunctionGradient(ProbabilisticModelWithGradient model,
                                                                  BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>> calculation) {
            return new LogLikelihoodFitnessFunctionGradient(model, calculation);

        }
    },
    MAP {
        @Override
        public FitnessFunction getFitnessFunction(ProbabilisticModel model,
                                                  BiConsumer<Map<VariableReference, DoubleTensor>, Double> calculation) {
            return new LogProbFitnessFunction(model, calculation);
        }

        @Override
        public FitnessFunctionGradient getFitnessFunctionGradient(ProbabilisticModelWithGradient model,
                                                                  BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>> calculation) {
            return new LogProbFitnessFunctionGradient(model, calculation);
        }
    };

    public abstract FitnessFunction getFitnessFunction(ProbabilisticModel model,
                                                       BiConsumer<Map<VariableReference, DoubleTensor>, Double> calculation);

    public FitnessFunction getFitnessFunction(ProbabilisticModel model) {
        return getFitnessFunction(model, null);
    }

    public abstract FitnessFunctionGradient getFitnessFunctionGradient(ProbabilisticModelWithGradient model,
                                                                       BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>> calculation);

    public FitnessFunctionGradient getFitnessFunctionGradient(ProbabilisticModelWithGradient model) {
        return getFitnessFunctionGradient(model, null);
    }

}
