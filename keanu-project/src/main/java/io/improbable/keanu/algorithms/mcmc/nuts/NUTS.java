package io.improbable.keanu.algorithms.mcmc.nuts;

import com.google.common.base.Preconditions;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.ProbabilisticModel;
import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.Statistics;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.mcmc.NetworkSamplesGenerator;
import io.improbable.keanu.algorithms.mcmc.SamplingAlgorithm;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.status.StatusBar;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.List;
import java.util.Map;

import static io.improbable.keanu.algorithms.mcmc.nuts.AdaptiveStepSize.findStartingStepSizeSimple;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.ones;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.zeros;
import static java.util.stream.Collectors.toMap;


/**
 * Algorithm 6: "No-U-Turn Sampler with Dual Averaging".
 * The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
 * https://arxiv.org/pdf/1111.4246.pdf
 */
@AllArgsConstructor
public class NUTS implements PosteriorSamplingAlgorithm {

    private static final int DEFAULT_ADAPT_COUNT = 1000;
    private static final double DEFAULT_TARGET_ACCEPTANCE_PROB = 0.8;

    private final Statistics statistics = new Statistics(Metrics.values());

    public static NUTSBuilder builder() {
        return new NUTSBuilder();
    }

    public enum Metrics {
        STEPSIZE, LOG_PROB, MEAN_TREE_ACCEPT, TREE_SIZE
    }

    @Getter
    private KeanuRandom random;

    //The number of samples for which the step size will be tuned. For the remaining samples
    //in which it is not tuned, the step size will be frozen to its last calculated value
    @Getter
    private int adaptCount;

    //The target acceptance probability, a suggested value of this is 0.65,
    //Beskos et al., 2010; Neal, 2011
    @Getter
    private double targetAcceptanceProb;

    //Determines whether the step size wil
    // l adapt during the first adaptCount samples
    private boolean adaptEnabled;

    //Sets the initial step size. If none is given then a heuristic will be used to determine a good step size.
    private Double initialStepSize;

    //The maximum tree size for the sampler. This controls how long a sample walk can be before it terminates. This
    //will set at a maximum approximately 2^treeSize number of logProb evaluations for a sample.
    private int maxTreeHeight;

    //Sets whether or not to save debug STATISTICS. The STATISTICS available are: Step size, Log Prob, Mean Tree Acceptance Prob, Tree Size.
    private boolean saveStatistics;

    /**
     * Sample from the posterior of a probabilistic model using the No-U-Turn-Sampling algorithm
     *
     * @param model                 the probabilistic model to sample from
     * @param variablesToSampleFrom the variables inside the probabilistic model to sample from
     * @return Samples taken with NUTS
     */
    @Override
    public NetworkSamples getPosteriorSamples(final ProbabilisticModel model,
                                              final List<? extends Variable> variablesToSampleFrom,
                                              final int sampleCount) {
        return generatePosteriorSamples(model, variablesToSampleFrom).generate(sampleCount);
    }

    @Override
    public NetworkSamplesGenerator generatePosteriorSamples(final ProbabilisticModel model,
                                                            final List<? extends Variable> fromVariables) {
        Preconditions.checkArgument(model instanceof ProbabilisticModelWithGradient, "NUTS requires a model on which gradients can be calculated.");
        return new NetworkSamplesGenerator(setupSampler((ProbabilisticModelWithGradient) model, fromVariables), StatusBar::new);
    }

    private NUTSSampler setupSampler(final ProbabilisticModelWithGradient model,
                                     final List<? extends Variable> sampleFromVariables) {

        Preconditions.checkArgument(!sampleFromVariables.isEmpty(), "List of variables to sample from is empty");

        final List<? extends Variable<DoubleTensor, ?>> latentVariables = model.getContinuousLatentVariables();

        Map<VariableReference, DoubleTensor> position = latentVariables.stream()
            .collect(toMap(Variable::getReference, Variable::getValue));

        double initialLogOfMasterP = model.logProb(position);

        Preconditions.checkArgument(
            !ProbabilityCalculator.isImpossibleLogProb(initialLogOfMasterP),
            "Sampler starting position is invalid. Please start from a non-zero probability position."
        );

        Map<? extends VariableReference, DoubleTensor> gradient = model.logProbGradients();

        Map<VariableReference, ?> startingSample = SamplingAlgorithm.takeSample(sampleFromVariables);

        double startingStepSize = (initialStepSize == null) ?
            findStartingStepSizeSimple(0.25, latentVariables) :
            initialStepSize;

        AdaptiveStepSize stepsize = new AdaptiveStepSize(
            startingStepSize,
            targetAcceptanceProb,
            adaptCount
        );

        Potential potential = new AdaptiveQuadraticPotential(zeros(position), ones(position), 10.0, adaptCount, random);

        Leapfrog startState = new Leapfrog(position, potential.random(), gradient, initialLogOfMasterP, potential);

        Proposal initialProposal = new Proposal(position, gradient, startingSample, startState.getEnergy(), 1.0, initialLogOfMasterP);

        Tree tree = new Tree(startState, initialProposal, 0.0, true, 0.0, 1, startState.getEnergy());

        return new NUTSSampler(
            sampleFromVariables,
            latentVariables,
            model,
            potential,
            adaptEnabled,
            stepsize,
            tree,
            maxTreeHeight,
            random,
            statistics,
            saveStatistics
        );
    }

    public Statistics getStatistics() {
        return statistics;
    }

    public static class NUTSBuilder {
        private KeanuRandom random = KeanuRandom.getDefaultRandom();
        private int adaptCount = DEFAULT_ADAPT_COUNT;
        private double targetAcceptanceProb = DEFAULT_TARGET_ACCEPTANCE_PROB;
        private boolean adaptEnabled = true;
        private Double initialStepSize = null;
        private int maxTreeHeight = 10;
        private boolean saveStatistics = false;

        NUTSBuilder() {
        }

        public NUTSBuilder random(KeanuRandom random) {
            this.random = random;
            return this;
        }

        public NUTSBuilder adaptCount(int adaptCount) {
            this.adaptCount = adaptCount;
            return this;
        }

        public NUTSBuilder targetAcceptanceProb(double targetAcceptanceProb) {
            this.targetAcceptanceProb = targetAcceptanceProb;
            return this;
        }

        public NUTSBuilder adaptEnabled(boolean adaptEnabled) {
            this.adaptEnabled = adaptEnabled;
            return this;
        }

        public NUTSBuilder initialStepSize(Double initialStepSize) {
            this.initialStepSize = initialStepSize;
            return this;
        }

        public NUTSBuilder maxTreeHeight(int maxTreeHeight) {
            this.maxTreeHeight = maxTreeHeight;
            return this;
        }

        public NUTSBuilder saveStatistics(boolean saveStatistics) {
            this.saveStatistics = saveStatistics;
            return this;
        }

        public NUTS build() {
            return new NUTS(random, adaptCount, targetAcceptanceProb, adaptEnabled, initialStepSize, maxTreeHeight, saveStatistics);
        }

        public String toString() {
            return "NUTS.NUTSBuilder(random=" + this.random + ", adaptCount=" + this.adaptCount + ", targetAcceptanceProb=" + this.targetAcceptanceProb + ", adaptEnabled=" + this.adaptEnabled + ", initialStepSize=" + this.initialStepSize + ", maxTreeHeight=" + this.maxTreeHeight + ", saveStatistics=" + this.saveStatistics + ")";
        }
    }
}
