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
import lombok.AccessLevel;
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
@AllArgsConstructor(access = AccessLevel.PRIVATE)
public class NUTS implements PosteriorSamplingAlgorithm {

    public static NUTSBuilder builder() {
        return new NUTSBuilder();
    }

    public enum Metrics {
        STEPSIZE, LOG_PROB, MEAN_TREE_ACCEPT, TREE_SIZE
    }

    @Getter
    private final KeanuRandom random;

    //The target acceptance probability, a suggested value of this is 0.65,
    //Beskos et al., 2010; Neal, 2011
    @Getter
    private final double targetAcceptanceProb;

    //The number of samples for which the step size and potential will be tuned. For the remaining samples
    //in which it is not tuned, the step size will be frozen to its last calculated value
    @Getter
    private final int adaptCount;

    //Determines whether the step size wil
    // l adapt during the first adaptCount samples
    private final boolean adaptStepSizeEnabled;

    //Sets the initial step size. If none is given then a heuristic will be used to determine a good step size.
    private final Double initialStepSize;

    private final boolean adaptPotentialEnabled;

    private final double maxEnergyChange;

    //The maximum tree size for the sampler. This controls how long a sample walk can be before it terminates. This
    //will set at a maximum approximately 2^treeSize number of logProb evaluations for a sample.
    private final int maxTreeHeight;

    //Sets whether or not to save debug STATISTICS. The STATISTICS available are: Step size, Log Prob, Mean Tree Acceptance Prob, Tree Size.
    private final boolean saveStatistics;

    private final Statistics statistics = new Statistics(Metrics.values());

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

        AdaptiveStepSize stepSize = new AdaptiveStepSize(
            startingStepSize,
            targetAcceptanceProb,
            adaptCount
        );

        Potential potential = new AdaptiveQuadraticPotential(zeros(position), ones(position), 10.0, adaptCount, 101, random);

        Proposal initialProposal = new Proposal(position, gradient, startingSample, initialLogOfMasterP);

        return new NUTSSampler(
            sampleFromVariables,
            model,
            adaptPotentialEnabled,
            potential,
            adaptStepSizeEnabled,
            stepSize,
            maxEnergyChange,
            initialProposal,
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

        private int adaptCount = 1000;
        private boolean adaptStepSizeEnabled = true;
        private Double initialStepSize = null;
        private boolean adaptPotentialEnabled = true;

        private double targetAcceptanceProb = 0.8;
        private double maxEnergyChange = 1000.0;
        private int maxTreeHeight = 10;

        private boolean saveStatistics = false;

        public NUTSBuilder random(KeanuRandom random) {
            this.random = random;
            return this;
        }

        public NUTSBuilder targetAcceptanceProb(double targetAcceptanceProb) {
            if (targetAcceptanceProb > 1.0 || targetAcceptanceProb < 0) {
                throw new IllegalArgumentException("Target acceptance probability must be between 0.0 and 1.");
            }
            this.targetAcceptanceProb = targetAcceptanceProb;
            return this;
        }

        public NUTSBuilder adaptStepSizeEnabled(boolean adaptEnabled) {
            this.adaptStepSizeEnabled = adaptEnabled;
            return this;
        }

        public NUTSBuilder adaptCount(int adaptCount) {
            if (adaptCount < 0) {
                throw new IllegalArgumentException("Adapt count must be greater than or equal to 0");
            }
            this.adaptCount = adaptCount;
            return this;
        }

        public NUTSBuilder initialStepSize(Double initialStepSize) {
            if (initialStepSize <= 0) {
                throw new IllegalArgumentException("Initial step size must be greater than 0");
            }
            this.initialStepSize = initialStepSize;
            return this;
        }

        public NUTSBuilder adaptPotentialEnabled(boolean adaptPotentialEnabled) {
            this.adaptPotentialEnabled = adaptPotentialEnabled;
            return this;
        }

        public NUTSBuilder maxEnergyChange(double maxEnergyChange) {
            if (maxEnergyChange <= 0) {
                throw new IllegalArgumentException("Max energy change must be greater than 0");
            }
            this.maxEnergyChange = maxEnergyChange;
            return this;
        }

        public NUTSBuilder maxTreeHeight(int maxTreeHeight) {
            if (maxTreeHeight <= 0) {
                throw new IllegalArgumentException("Max tree height must be greater than 0");
            }
            this.maxTreeHeight = maxTreeHeight;
            return this;
        }

        public NUTSBuilder saveStatistics(boolean saveStatistics) {
            this.saveStatistics = saveStatistics;
            return this;
        }

        public NUTS build() {
            return new NUTS(random, targetAcceptanceProb, adaptCount, adaptStepSizeEnabled, initialStepSize,
                adaptPotentialEnabled, maxEnergyChange, maxTreeHeight, saveStatistics);
        }

        public String toString() {
            return "NUTS.NUTSBuilder(random=" + this.random + ", adaptCount=" + this.adaptCount +
                ", targetAcceptanceProb=" + this.targetAcceptanceProb + ", adaptStepSizeEnabled=" +
                this.adaptStepSizeEnabled + ", initialStepSize=" + this.initialStepSize + ", maxTreeHeight=" +
                this.maxTreeHeight + ", saveStatistics=" + this.saveStatistics + ")";
        }
    }
}
