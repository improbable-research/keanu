package io.improbable.keanu.network;

import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * An implementation of {@link ProbabilisticModelWithGradient} that is backed by a {@link BayesianNetwork}
 */
public class KeanuProbabilisticModelWithGradient extends KeanuProbabilisticModel implements ProbabilisticModelWithGradient {

    private final LogProbGradientCalculator logProbGradientCalculator;
    private final LogProbGradientCalculator logLikelihoodGradientCalculator;

    public KeanuProbabilisticModelWithGradient(BayesianNetwork bayesianNetwork) {
        super(bayesianNetwork);

        List<DoubleVertex> continuousLatentVertices = bayesianNetwork.getContinuousLatentVertices();

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
    public Map<VariableReference, DoubleTensor> logProbGradients(Map<VariableReference, ?> inputs) {
        return gradients(inputs, logProbGradientCalculator);
    }

    @Override
    public Map<VariableReference, DoubleTensor> logProbGradients() {
        return logProbGradients(null);
    }

    @Override
    public Map<VariableReference, DoubleTensor> logLikelihoodGradients(Map<VariableReference, ?> inputs) {
        return gradients(inputs, logLikelihoodGradientCalculator);
    }

    @Override
    public Map<VariableReference, DoubleTensor> logLikelihoodGradients() {
        return logLikelihoodGradients(null);
    }

    private Map gradients(Map<VariableReference, ?> inputs, LogProbGradientCalculator gradientCalculator) {
        if (inputs != null && !inputs.isEmpty()) {
            cascadeValues(inputs);
        }

        return gradientCalculator.getJointLogProbGradientWrtLatents();
    }

}
