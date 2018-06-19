package examples;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.ProbabilisticInteger;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.jetbrains.annotations.NotNull;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

/**
 * What's the likelihood of a student cheating in a test, given obfuscated survey data?
 *
 * http://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter2_MorePyMC/Ch2_MorePyMC_PyMC3.ipynb#Example:-Cheating-among-students
 */
public class CheatingStudents {

    public static CheatingStudentsPosteriors run() {
        UniformVertex pCheating = new UniformVertex(0., 1.);
        DoubleVertex skew = pCheating.times(.5).plus(.25);

        BinomialVertex reportedCheating = new BinomialVertex(new int[] {1, 1}, new ConstantIntegerVertex(100), skew);
        reportedCheating.observe(35);

        BayesianNetwork net = new BayesianNetwork(pCheating.getConnectedGraph());

        NetworkSamples networkSamples = MetropolisHastings.getPosteriorSamples(net, net.getLatentVertices(), 3000)
                .drop(500).downSample(1);

        CheatingStudentsPosteriors out = new CheatingStudentsPosteriors();
        out.freqCheating = networkSamples.get(pCheating).asList().stream().map(dT -> dT.scalar()).collect(Collectors.toList());
        out.freqCheatingMode = networkSamples.get(pCheating).getMode().scalar();
        return out;
    }

    static class BinomialVertex extends ProbabilisticInteger {
        private IntegerVertex n;
        private DoubleVertex p;

        public BinomialVertex(int[] shape, IntegerVertex n, DoubleVertex p) {
            checkTensorsMatchNonScalarShapeOrAreScalar(shape, p.getShape());
            checkTensorsMatchNonScalarShapeOrAreScalar(shape, n.getShape());

            setParents(n, p);
            this.n = n;
            this.p = p;
            setValue(IntegerTensor.placeHolder(shape));
        }

        @Override
        public double logPmf(IntegerTensor value) {
            return getBinomialDistribution().logProbability(value.scalar());
        }

        @Override
        public Map<Long, DoubleTensor> dLogPmf(IntegerTensor value) {
            throw new UnsupportedOperationException();
        }

        @Override
        public IntegerTensor sample(KeanuRandom random) {
            return IntegerTensor.scalar(getBinomialDistribution().sample());
        }

        @NotNull
        private BinomialDistribution getBinomialDistribution() {
            return new BinomialDistribution(n.getValue().scalar(), p.getValue().scalar());
        }
    }

    public static class CheatingStudentsPosteriors {
        private List<Double> freqCheating;
        private double freqCheatingMode;

        public List<Double> getFreqCheating() {
            return freqCheating;
        }

        public double getFreqCheatingMode() {
            return freqCheatingMode;
        }
    }
}
