package io.improbable.keanu.vertices.utility;

import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import java.io.PrintStream;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.atLeast;
import static org.mockito.Mockito.verify;

public class PrintVertexTest {
    @Mock
    PrintStream printStream;

    @Before
    public void setup() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testWhenCreatedThenParentIsSet() {
        final DoubleVertex parent = new ConstantDoubleVertex(30);
        final PrintVertex pv = new PrintVertex(parent);

        assertThat(pv.getParents()).isEqualTo(ImmutableSet.of(parent));
    }

    @Test(expected = NullPointerException.class)
    public void testWhenCreatedWithNullThenFails() {
        new PrintVertex(null);
    }

    @Test(expected = NullPointerException.class)
    public void testWhenPrintStreamNullThenFails() {
        final DoubleVertex parent = new ConstantDoubleVertex(30);
        new PrintVertex(parent, null);
    }

    @Test
    public void testWhenParentHasLabelThenItIsPrinted() {
        final DoubleVertex parent = new ConstantDoubleVertex(30);
        String label = "my vertex";
        parent.setLabel(label);
        final PrintVertex pv = new PrintVertex(parent, printStream);

        pv.calculate();

        final String expected = "Calculated Vertex: (label: my vertex, data: {\ndata = [30.0]\nshape = []\n})";
        verify(printStream).println(expected);
    }

    @Test
    public void testWhenParentHasNoLabelThenPlaceholderPrinted() {
        final DoubleVertex parent = new ConstantDoubleVertex(30);
        final PrintVertex pv = new PrintVertex(parent, printStream);

        pv.calculate();

        final String expected = "Calculated Vertex: (label: <no label>, data: {\ndata = [30.0]\nshape = []\n})";
        verify(printStream).println(expected);
    }

    @Test
    public void testWhenRunningMetropolisHastingsThenSamplesArePrinted() {
        final UniformVertex temperature = new UniformVertex(20., 30.);
        final PrintVertex pv = new PrintVertex(temperature, printStream);
        final BayesianNetwork bayesNet = new BayesianNetwork(pv.getConnectedGraph());

        final int nSamples = 1000000;
        MetropolisHastings
                .withDefaultConfig()
                .getPosteriorSamples(bayesNet, bayesNet.getLatentVertices(),nSamples);

        verify(printStream, atLeast(nSamples)).println(any(String.class));
    }
}
