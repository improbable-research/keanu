package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;

import java.util.List;
import java.util.Map;
import java.util.Set;

public class KeanuProbabilisticModelWithGradient extends KeanuProbabilisticModel implements ProbabilisticModelWithGradient {

    private LogProbGradientCalculator logProbGradientCalculator;
    private LogProbGradientCalculator logLikelihoodGradientCalculator;

    public KeanuProbabilisticModelWithGradient(BayesianNetwork bayesianNetwork) {
        super(bayesianNetwork);

        List<Vertex<DoubleTensor>> continuousLatentVertices = bayesianNetwork.getContinuousLatentVertices();

        this.logProbGradientCalculator = new LogProbGradientCalculator(
            bayesianNetwork.getLatentOrObservedVertices(),
            continuousLatentVertices
        );

        this.logLikelihoodGradientCalculator = new LogProbGradientCalculator(
            bayesianNetwork.getObservedVertices(),
            continuousLatentVertices
        );
    }

    public KeanuProbabilisticModelWithGradient(Set<Vertex> variables) {
        this(new BayesianNetwork(variables));
    }

    @Override
    public Map<? extends VariableReference, DoubleTensor> logProbGradients(Map<VariableReference, ?> inputs) {
        return gradients(inputs, logProbGradientCalculator);
    }

    @Override
    public Map<? extends VariableReference, DoubleTensor> logProbGradients() {
        return logProbGradients(null);
    }

    @Override
    public Map<? extends VariableReference, DoubleTensor> logLikelihoodGradients(Map<VariableReference, ?> inputs) {
        return gradients(inputs, logLikelihoodGradientCalculator);
    }

    @Override
    public Map<? extends VariableReference, DoubleTensor> logLikelihoodGradients() {
        return logLikelihoodGradients(null);
    }

    private Map<? extends VariableReference, DoubleTensor> gradients(Map<VariableReference, ?> inputs, LogProbGradientCalculator gradientCalculator) {
        if (inputs != null && !inputs.isEmpty()) {
            cascadeValues(inputs);
        }

        return gradientCalculator.getJointLogProbGradientWrtLatents();
    }

}
