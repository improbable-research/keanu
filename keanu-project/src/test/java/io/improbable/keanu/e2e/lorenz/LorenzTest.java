package io.improbable.keanu.e2e.lorenz;

import io.improbable.keanu.algorithms.variational.TensorGradientOptimizer;
import io.improbable.keanu.network.BayesNetTensorAsContinuous;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantDoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorGaussianVertex;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.assertTrue;

public class LorenzTest {
    private final Logger log = LoggerFactory.getLogger(LorenzTest.class);

    @Test
    public void convergesOnLorenz() {

        double[] priorMu = new double[]{3, 3, 3};
        double error = Double.MAX_VALUE;
        double convergedError = 0.01;
        int windowSize = 8;
        int window = 0;
        int maxWindows = 100;

        LorenzModel model = new LorenzModel();
        List<LorenzModel.Coordinates> observed = model.runModel(windowSize * maxWindows);

        while (error > convergedError && window < maxWindows) {

            TensorGaussianVertex xt0 = new TensorGaussianVertex(priorMu[0], 1.0);
            TensorGaussianVertex yt0 = new TensorGaussianVertex(priorMu[1], 1.0);
            TensorGaussianVertex zt0 = new TensorGaussianVertex(priorMu[2], 1.0);

            List<List<DoubleTensorVertex>> graphTimeSteps = new ArrayList<>();
            graphTimeSteps.add(Arrays.asList(xt0, yt0, zt0));

            //Build graph
            for (int i = 1; i < windowSize; i++) {
                List<DoubleTensorVertex> ti = graphTimeSteps.get(i - 1);
                List<DoubleTensorVertex> tiPlus1 = addTime(
                    ti.get(0), ti.get(1), ti.get(2),
                    LorenzModel.timeStep, LorenzModel.sigma, LorenzModel.rho, LorenzModel.beta
                );
                graphTimeSteps.add(tiPlus1);
            }

            xt0.setAndCascade(priorMu[0]);
            yt0.setAndCascade(priorMu[1]);
            zt0.setAndCascade(priorMu[2]);

            //Apply observations
            for (int i = 0; i < graphTimeSteps.size(); i++) {

                int t = window * (windowSize - 1) + i;

                List<DoubleTensorVertex> timeSlice = graphTimeSteps.get(i);
                DoubleTensorVertex xt = timeSlice.get(0);

                TensorGaussianVertex observedXt = new TensorGaussianVertex(xt, new ConstantDoubleTensorVertex(1.0));
                observedXt.observe(observed.get(t).x);
            }

            BayesNetTensorAsContinuous net = new BayesNetTensorAsContinuous(xt0.getConnectedGraph());

            TensorGradientOptimizer graphOptimizer = new TensorGradientOptimizer(net);

            graphOptimizer.maxAPosteriori(1000);

            List<DoubleTensor> posterior = getTimeSliceValues(graphTimeSteps, windowSize - 1);

            int postT = (window + 1) * (windowSize - 1);
            LorenzModel.Coordinates actualAtPostT = observed.get(postT);

            error = Math.sqrt(
                Math.pow(actualAtPostT.x - posterior.get(0).scalar(), 2) +
                    Math.pow(actualAtPostT.y - posterior.get(1).scalar(), 2) +
                    Math.pow(actualAtPostT.z - posterior.get(2).scalar(), 2)
            );

            log.info("Error: " + error);

            priorMu = new double[]{posterior.get(0).scalar(), posterior.get(1).scalar(), posterior.get(2).scalar()};
            window++;
        }

        assertTrue(error <= convergedError);
    }

    private List<DoubleTensorVertex> addTime(DoubleTensorVertex xt,
                                             DoubleTensorVertex yt,
                                             DoubleTensorVertex zt,
                                             double timestep,
                                             double sigma,
                                             double rho,
                                             double beta) {

        DoubleTensorVertex rhov = new ConstantDoubleTensorVertex(rho);

        DoubleTensorVertex xtplus1 = xt.multiply(1 - timestep * sigma).plus(yt.multiply(timestep * sigma));

        DoubleTensorVertex ytplus1 = yt.multiply(1 - timestep).plus(xt.multiply(rhov.minus(zt)).multiply(timestep));

        DoubleTensorVertex ztplus1 = zt.multiply(1 - timestep * beta).plus(xt.multiply(yt).multiply(timestep));

        return Arrays.asList(xtplus1, ytplus1, ztplus1);
    }

    private List<DoubleTensor> getTimeSliceValues(List<List<DoubleTensorVertex>> graphTimeSteps, int time) {
        List<DoubleTensorVertex> slice = graphTimeSteps.get(time);

        return slice.stream()
            .map(Vertex::getValue)
            .collect(Collectors.toList());
    }

}
