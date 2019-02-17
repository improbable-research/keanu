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

import static io.improbable.keanu.algorithms.mcmc.nuts.AdaptiveStepSize.findStartingStepSize;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.ones;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.zeros;
import static java.util.stream.Collectors.toMap;


/**
 * NUTS with multinomial sampling
 * <p>
 * References:
 * <p>
 * Algorithm 6: "No-U-Turn Sampler with Dual Averaging".
 * The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
 * https://arxiv.org/pdf/1111.4246.pdf
 * <p>
 * A Conceptual Introduction to
 * Hamiltonian Monte Carlo by Michael Betancourt
 * https://arxiv.org/pdf/1701.02434.pdf
 */
@AllArgsConstructor(access = AccessLevel.PRIVATE)
public class NUTS implements PosteriorSamplingAlgorithm {

    public static NUTSBuilder builder() {
        return new NUTSBuilder();
    }

    public enum Metrics {
        STEPSIZE, LOG_PROB, MEAN_TREE_ACCEPT, TREE_SIZE, DIVERGENT_SAMPLE
    }

    @Getter
    private final KeanuRandom random;

    @Getter
    private final double targetAcceptanceProb;

    @Getter
    private final long adaptCount;

    private final boolean adaptStepSizeEnabled;

    private final Double initialStepSize;

    private final Potential potential;

    private final boolean adaptPotentialEnabled;

    private final double maxEnergyChange;

    private final int maxTreeHeight;

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
            findStartingStepSize(0.25, latentVariables) :
            initialStepSize;

        AdaptiveStepSize stepSize = new AdaptiveStepSize(
            startingStepSize,
            targetAcceptanceProb,
            adaptCount
        );

        potential.initialize(zeros(position), ones(position));
        Proposal initialProposal = new Proposal(position, gradient, startingSample, initialLogOfMasterP);

        return new NUTSSampler(
            sampleFromVariables,
            model,
            adaptPotentialEnabled,
            potential,
            adaptStepSizeEnabled,
            stepSize,
            adaptCount,
            maxEnergyChange,
            maxTreeHeight,
            initialProposal,
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

        private long adaptCount = 1000;
        private boolean adaptStepSizeEnabled = true;
        private Double initialStepSize = null;

        private Potential potential = new AdaptiveQuadraticPotential(10.0, 100);
        private boolean adaptPotentialEnabled = true;

        private double targetAcceptanceProb = 0.8;
        private double maxEnergyChange = 1000.0;
        private int maxTreeHeight = 10;

        private boolean saveStatistics = false;

        public NUTSBuilder random(KeanuRandom random) {
            this.random = random;
            return this;
        }

        /**
         * @param targetAcceptanceProb The targeted acceptance rate for the step size adaption. This defaults to 0.8.
         * @return the builder for NUTS
         */
        public NUTSBuilder targetAcceptanceProb(double targetAcceptanceProb) {
            if (targetAcceptanceProb > 1.0 || targetAcceptanceProb < 0) {
                throw new IllegalArgumentException("Target acceptance probability must be between 0.0 and 1.");
            }
            this.targetAcceptanceProb = targetAcceptanceProb;
            return this;
        }

        /**
         * @param adaptEnabled Set to true if step size adaption is wanted.
         * @return the builder for NUTS
         */
        public NUTSBuilder adaptStepSizeEnabled(boolean adaptEnabled) {
            this.adaptStepSizeEnabled = adaptEnabled;
            return this;
        }

        /**
         * @param adaptCount The number of samples for which the step size and potential will be tuned. For the
         *                   remaining samples in which it is not tuned, the step size will be frozen to its last
         *                   calculated value.
         * @return the builder for NUTS
         */
        public NUTSBuilder adaptCount(long adaptCount) {
            if (adaptCount < 0) {
                throw new IllegalArgumentException("Adapt count must be greater than or equal to 0");
            }
            this.adaptCount = adaptCount;
            return this;
        }

        /**
         * @param initialStepSize The initial step size. If this is null then a step size will be calculated using
         *                        heuristics.
         * @return the builder for NUTS
         */
        public NUTSBuilder initialStepSize(Double initialStepSize) {
            if (initialStepSize <= 0) {
                throw new IllegalArgumentException("Initial step size must be greater than 0");
            }
            this.initialStepSize = initialStepSize;
            return this;
        }

        /**
         * @param adaptPotentialEnabled Set to true if adapting of the mass matrix is wanted.
         * @return the builder for NUTS
         */
        public NUTSBuilder adaptPotentialEnabled(boolean adaptPotentialEnabled) {
            this.adaptPotentialEnabled = adaptPotentialEnabled;
            return this;
        }

        /**
         * @param potential provides mass in velocity and energy calculations
         * @return the builder for NUTS
         */
        public NUTSBuilder potential(Potential potential) {
            this.potential = potential;
            return this;
        }

        /**
         * @param maxEnergyChange The maximum energy change for a step to be considered divergent.
         * @return the builder for NUTS
         */
        public NUTSBuilder maxEnergyChange(double maxEnergyChange) {
            if (maxEnergyChange <= 0) {
                throw new IllegalArgumentException("Max energy change must be greater than 0");
            }
            this.maxEnergyChange = maxEnergyChange;
            return this;
        }


        /**
         * @param maxTreeHeight The maximum tree size for the sampler. This controls how long a sample walk can
         *                      be before it terminates. This will set at a maximum approximately 2^treeSize
         *                      number of logProb evaluations for a sample.
         * @return the builder for NUTS
         */
        public NUTSBuilder maxTreeHeight(int maxTreeHeight) {
            if (maxTreeHeight <= 0) {
                throw new IllegalArgumentException("Max tree height must be greater than 0");
            }
            this.maxTreeHeight = maxTreeHeight;
            return this;
        }

        /**
         * @param saveStatistics Set to true if sampling statistics are wanted. Sets whether or not to save
         *                       debug STATISTICS. The STATISTICS available are: Step size, Log Prob,
         *                       Mean Tree Acceptance Prob, Tree Size.
         * @return the builder for NUTS
         */
        public NUTSBuilder saveStatistics(boolean saveStatistics) {
            this.saveStatistics = saveStatistics;
            return this;
        }

        public NUTS build() {
            return new NUTS(random, targetAcceptanceProb, adaptCount, adaptStepSizeEnabled, initialStepSize,
                potential, adaptPotentialEnabled, maxEnergyChange, maxTreeHeight, saveStatistics);
        }

        public String toString() {
            return "NUTS.NUTSBuilder(random=" + this.random + ", adaptCount=" + this.adaptCount +
                ", targetAcceptanceProb=" + this.targetAcceptanceProb + ", adaptStepSizeEnabled=" +
                this.adaptStepSizeEnabled + ", initialStepSize=" + this.initialStepSize + ", maxTreeHeight=" +
                this.maxTreeHeight + ", saveStatistics=" + this.saveStatistics + ")";
        }
    }
}
