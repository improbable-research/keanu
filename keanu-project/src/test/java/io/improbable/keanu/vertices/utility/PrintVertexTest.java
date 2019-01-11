package io.improbable.keanu.vertices.utility;

import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import java.io.PrintStream;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.atLeast;
import static org.mockito.Mockito.verify;

@RunWith(MockitoJUnitRunner.class)
public class PrintVertexTest {
    @Mock
    PrintStream printStream;

    @Before
    public void setup() {
        PrintVertex.setPrintStream(printStream);
    }

    @Test
    public void whenCreatedThenParentIsSet() {
        final Vertex<DoubleTensor> parent = new ConstantDoubleVertex(30);

        final PrintVertex<DoubleTensor> sut = new PrintVertex<>(parent);

        assertThat(sut.getParents(), equalTo(ImmutableSet.of(parent)));
    }

    @Test(expected = NullPointerException.class)
    public void whenCreatedWithNullParentThenFails() {
        new PrintVertex<>(null);
    }

    @Test(expected = NullPointerException.class)
    public void whenCreatedWithNullPrefixParentThenFails() {
        final DoubleVertex parent = new ConstantDoubleVertex(30);
        new PrintVertex<>(parent, null, false);
    }

    @Test(expected = NullPointerException.class)
    public void whenPrintStreamIsNullThenFails() {
        PrintVertex.setPrintStream(null);
    }

    @Test
    public void whenOnlyParentIsSpecifiedThenDefaultPresentationIsUsed() {
        final Vertex<DoubleTensor> parent = new ConstantDoubleVertex(30);

        final PrintVertex<DoubleTensor> sut = new PrintVertex<>(parent);

        sut.calculate();
        final String expected = "Calculated Vertex:\n" +
            "{\n" +
            "data = [30.0]\n" +
            "shape = []\n" +
            "}\n";
        verify(printStream).print(expected);
    }

    @Test
    public void whenPrefixIsSuppliedThenItIsUsedInPresentation() {
        final Vertex<DoubleTensor> parent = new ConstantDoubleVertex(30);

        final PrintVertex<DoubleTensor> sut = new PrintVertex<>(parent, "my vertex\n", false);

        sut.calculate();
        final String expected = "my vertex\n";
        verify(printStream).print(expected);
    }

    @Test
    public void whenPrefixAndDataFlagAreSuppliedThenTheyAreUsedInPresentation() {
        final Vertex<DoubleTensor> parent = new ConstantDoubleVertex(30);

        final PrintVertex<DoubleTensor> sut = new PrintVertex<>(parent, "my vertex\n", true);

        sut.calculate();
        final String expected = "my vertex\n" +
            "{\n" +
            "data = [30.0]\n" +
            "shape = []\n" +
            "}\n";
        verify(printStream).print(expected);
    }

    @Test
    public void whenRunningMetropolisHastingsThenSamplesArePrinted() {
        final Vertex<DoubleTensor> temperature = new UniformVertex(20., 30.);

        new PrintVertex<>(temperature);

        final BayesianNetwork bayesNet = new BayesianNetwork(temperature.getConnectedGraph());

        final int nSamples = 10000;
        MetropolisHastings
            .withDefaultConfig()
            .getPosteriorSamples(bayesNet, bayesNet.getLatentVertices(), nSamples);

        verify(printStream, atLeast(nSamples)).print(any(String.class));
    }
}
