package io.improbable.keanu.network;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

import static junit.framework.TestCase.assertEquals;

public class KeanuProbabilisticModelWithGradientTest {

    GaussianVertex A;
    GaussianVertex B;
    BernoulliVertex C;

    KeanuProbabilisticModelWithGradient model;

    @Before
    public void setup() {
        A = new GaussianVertex(0.0, 1.0);
        B = new GaussianVertex(0.0, 1.0);

        A.setValue(0.5);
        B.setValue(0.25);
        C = new BernoulliVertex(A.times(B));
        C.observe(true);

        model = new KeanuProbabilisticModelWithGradient(C.getConnectedGraph());
    }

    @Test
    public void canCalculateLogProbGradient() {

        Map<? extends VariableReference, DoubleTensor> logLikelihoodGradients = model.logLikelihoodGradients();

        LogProbGradientCalculator logLikelihoodGradientCalculator = new LogProbGradientCalculator(
            Collections.singletonList(C), Arrays.asList(A, B)
        );

        canCalculateLogProbGradient(logLikelihoodGradientCalculator, logLikelihoodGradients);
    }

    @Test
    public void canCalculateLogLikelihoodGradient() {

        Map<? extends VariableReference, DoubleTensor> logProbGradients = model.logProbGradients();
        LogProbGradientCalculator logProbGradientCalculator = new LogProbGradientCalculator(
            Arrays.asList(C, A, B), Arrays.asList(A, B)
        );

        canCalculateLogProbGradient(logProbGradientCalculator, logProbGradients);
    }

    public void canCalculateLogProbGradient(LogProbGradientCalculator gradientCalculator,
                                            Map<? extends VariableReference, DoubleTensor> actualGradients) {

        Map<VertexId, DoubleTensor> expectedGradients = gradientCalculator.getJointLogProbGradientWrtLatents();

        DoubleTensor dLogProbWrtA = actualGradients.get(A.getId());
        DoubleTensor dLogProbWrtB = actualGradients.get(B.getId());

        assertEquals(expectedGradients.get(A.getId()), dLogProbWrtA);
        assertEquals(expectedGradients.get(B.getId()), dLogProbWrtB);
    }

}
