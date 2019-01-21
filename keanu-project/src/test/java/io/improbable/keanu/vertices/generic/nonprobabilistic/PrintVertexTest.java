package io.improbable.keanu.vertices.generic.nonprobabilistic;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import io.improbable.keanu.algorithms.mcmc.KeanuMetropolisHastings;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticModel;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.io.ProtobufLoader;
import io.improbable.keanu.util.io.ProtobufSaver;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.UnaryOpVertex;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.samePropertyValuesAs;
import static org.hamcrest.core.IsInstanceOf.instanceOf;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.atLeast;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@RunWith(MockitoJUnitRunner.class)
public class PrintVertexTest {
    @Rule
    public ExpectedException thrown = ExpectedException.none();

    @Mock
    PrintStream printStream;

    @Mock
    DoubleVertex parent;

    @Before
    public void setup() {
        when(parent.getValue()).thenReturn(DoubleTensor.ONE_SCALAR);
        PrintVertex.setPrintStream(printStream);
    }

    @Test
    public void whenCreatedThenParentIsSet() {
        final PrintVertex<DoubleTensor> sut = new PrintVertex<>(parent);
        assertThat(sut.getParents(), equalTo(ImmutableSet.of(parent)));
    }

    @Test
    public void whenCreatedWithNullParentThenFails() {
        thrown.expect(NullPointerException.class);
        new PrintVertex<>(null);
    }

    @Test
    public void whenCreatedWithNullPrefixParentThenFails() {
        thrown.expect(NullPointerException.class);
        new PrintVertex<>(parent, null, false);
    }

    @Test
    public void whenPrintStreamIsNullThenFails() {
        thrown.expect(NullPointerException.class);
        PrintVertex.setPrintStream(null);
    }

    @Test
    public void whenOnlyParentIsSpecifiedThenDefaultPresentationIsUsed() {
        final PrintVertex<DoubleTensor> sut = new PrintVertex<>(parent);

        sut.calculate();
        final String expected = "Calculated Vertex:\n" +
            "{\n" +
            "data = [1.0]\n" +
            "shape = []\n" +
            "}\n";
        verify(printStream).print(expected);
    }

    @Test
    public void whenPrintVertexHasAChildThenChildCanGetGrandParentsValue() {
        // this use case is quite unlikely, the Print vertex is meant to hang off its parent but not have any children
        final Vertex<DoubleTensor> realParent = new ConstantDoubleVertex(30);
        final PrintVertex<DoubleTensor> sut = new PrintVertex<>(realParent);

        final UnaryOpVertex<DoubleTensor, DoubleTensor> child = new PlusOneOp(new long[]{1, 1}, sut);

        assertThat(sut.getValue().scalar(), equalTo(30.0));
        assertThat(child.getValue().scalar(), equalTo(31.0));
    }

    @Test
    public void whenSampleIsCalledThenKeanuRandomIsPassedToParentSample() {
        final KeanuRandom random = mock(KeanuRandom.class);

        final PrintVertex<DoubleTensor> sut = new PrintVertex<>(parent);
        sut.sample(random);
        verify(parent).sample(random);
    }

    @Test
    public void whenPrefixIsSuppliedThenItIsUsedInPresentation() {
        final PrintVertex<DoubleTensor> sut = new PrintVertex<>(parent, "my vertex\n", false);

        sut.calculate();
        final String expected = "my vertex\n";
        verify(printStream).print(expected);
    }

    @Test
    public void whenPrefixAndDataFlagAreSuppliedThenTheyAreUsedInPresentation() {
        final PrintVertex<DoubleTensor> sut = new PrintVertex<>(parent, "my vertex\n", true);

        sut.calculate();
        final String expected = "my vertex\n" +
            "{\n" +
            "data = [1.0]\n" +
            "shape = []\n" +
            "}\n";
        verify(printStream).print(expected);
    }

    @Test
    public void whenRunningMetropolisHastingsThenSamplesArePrinted() {
        final Vertex<DoubleTensor> temperature = new UniformVertex(20., 30.);

        new PrintVertex<>(temperature);

        KeanuProbabilisticModel model = new KeanuProbabilisticModel(temperature.getConnectedGraph());

        final int nSamples = 100;
        KeanuMetropolisHastings
            .withDefaultConfigFor(model)
            .getPosteriorSamples(model, model.getLatentVariables(), nSamples);

        verify(printStream, atLeast(nSamples)).print(anyString());
    }

    @Test
    public void whenPrintIsCalledAddsPrintVertexAsChild() {
        final UniformVertex A = new UniformVertex(0, 1);
        A.print();
        final Vertex printVertex = Iterables.getOnlyElement(A.getChildren());
        assertThat(printVertex, instanceOf(PrintVertex.class));
    }

    @Test
    public void whenPrintIsCalledWithOptionsThenOptionsArePassedToPrintVertex() {
        final PrintStream printStream = mock(PrintStream.class);
        PrintVertex.setPrintStream(printStream);

        final DoubleVertex A = new ConstantDoubleVertex(42);
        A.print("testprefix", true);

        final Vertex printVertex = Iterables.getOnlyElement(A.getChildren());
        printVertex.getValue();

        final String expectedOutput = "testprefix{\n" +
            "data = [42.0]\n" +
            "shape = []\n" +
            "}\n";

        verify(printStream).print(expectedOutput);
    }

    @Test
    public void whenSavingAndLoadingThenVertexStateIsRestored() throws IOException {
        final Vertex<DoubleTensor> realParent = new ConstantDoubleVertex(30);
        final PrintVertex<DoubleTensor> sut = new PrintVertex<>(realParent, "my vertex\n", true);

        final BayesianNetwork net = new BayesianNetwork(sut.getConnectedGraph());
        final BayesianNetwork readNet = saveLoad(net);

        assertThat(readNet.getAllVertices(), samePropertyValuesAs(net.getAllVertices()));
    }

    private BayesianNetwork saveLoad(final BayesianNetwork net) throws IOException {
        final ByteArrayOutputStream output = new ByteArrayOutputStream();

        final ProtobufSaver protobufSaver = new ProtobufSaver(net);
        protobufSaver.save(output, true);
        assertThat(output.size(), greaterThan(0));
        final ByteArrayInputStream input = new ByteArrayInputStream(output.toByteArray());

        final ProtobufLoader loader = new ProtobufLoader();
        return loader.loadNetwork(input);
    }

    private static class PlusOneOp extends UnaryOpVertex<DoubleTensor, DoubleTensor> implements NonSaveableVertex {
        public PlusOneOp(final long[] shape, final Vertex<DoubleTensor> inputVertex) {
            super(shape, inputVertex);
        }

        @Override
        protected DoubleTensor op(final DoubleTensor a) {
            return a.plus(1);
        }
    }
}
