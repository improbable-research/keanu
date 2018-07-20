package io.improbable.keanu.e2e.bm4h;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.BinomialVertex;

public class CheatingStudents {

    public static void main(String[] args) {
        CheatingStudents model = new CheatingStudents(100, 35);
        model.run(10000, 2);
        System.out.println(
            model.posteriorSamples
                .getDoubleTensorSamples(model.p)
                .getAverages()
                .scalar()
        );
    }

    CheatingStudents(int nStudents, int nYesAnswers) {
        this.nStudents = nStudents;
        this.nYesAnswers = nYesAnswers;
    }

    int nStudents;
    int nYesAnswers;
    UniformVertex p;
    NetworkSamples posteriorSamples;

    private BayesianNetwork buildNetwork1() {
        p = new UniformVertex(0.0, 1.0);
        BoolVertex studentCheated = new Flip(new int[]{1, nStudents}, p);
        BoolVertex answerIsTrue = new Flip(new int[]{1, nStudents}, 0.5);
        BoolVertex randomAnswer = new Flip(new int[]{1, nStudents}, 0.5);

        DoubleVertex answer = If.isTrue(answerIsTrue)
            .then(
                If.isTrue(studentCheated)
                    .then(1)
                    .orElse(0)
            ).orElse(
                If.isTrue(randomAnswer)
                    .then(1)
                    .orElse(0)
            );

        DoubleVertex answerTotal = new GaussianVertex(answer.sum(), 1);
        answerTotal.observe(nYesAnswers);

        return new BayesianNetwork(answerTotal.getConnectedGraph());
    }

    private BayesianNetwork buildNetwork2() {
        p = new UniformVertex(0.0, 1.0);
        DoubleVertex pYesAnswer = p.times(0.5).plus(0.25);
        IntegerVertex answerTotal = new BinomialVertex(pYesAnswer, nStudents);
        answerTotal.observe(nYesAnswers);

        return new BayesianNetwork(answerTotal.getConnectedGraph());
    }


    public void run(int nSamples, int version) {
        BayesianNetwork net;
        if (version == 1) {
            net = buildNetwork1();
        } else {
            net = buildNetwork2();
        }
        System.out.println(net.getLatentVertices());
        net.probeForNonZeroProbability(100000);
        posteriorSamples = MetropolisHastings.getPosteriorSamples(
            net,
            net.getLatentVertices(),
            nSamples
        ).drop(nSamples / 10).downSample(5);
    }

}
