package io.improbable.keanu.util.io;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.mir.MIR;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;

public class MIRTest {

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    @Test
    public void youCanSaveAndLoadANetworkWithValues() throws IOException {
        final String gaussianLabel = "Gaussian";
        DoubleVertex mu1 = new ConstantDoubleVertex(new double[]{3.0, 1.0});
        DoubleVertex mu2 = new ConstantDoubleVertex(new double[]{5.0, 6.0});
        DoubleVertex finalMu = new ConcatenationVertex(0, mu1, mu2);
        DoubleVertex gaussianVertex = new GaussianVertex(finalMu, 1.0);
        gaussianVertex.setLabel(gaussianLabel);
        BayesianNetwork net = new BayesianNetwork(gaussianVertex.getConnectedGraph());
        ByteArrayOutputStream output = new ByteArrayOutputStream();

        MIRSaver mirSaver = new MIRSaver(net);
        mirSaver.save(output, true);
        assertThat(output.size(), greaterThan(0));
        ByteArrayInputStream input = new ByteArrayInputStream(output.toByteArray());

        MIRLoader loader = new MIRLoader();
        BayesianNetwork readNet = loader.loadNetwork(input);

        assertThat(readNet.getLatentVertices().size(), is(1));
        assertThat(readNet.getLatentVertices().get(0), instanceOf(GaussianVertex.class));
        GaussianVertex latentGaussianVertex = (GaussianVertex) readNet.getLatentVertices().get(0);
        GaussianVertex labelGaussianVerted = (GaussianVertex) readNet.getVertexByLabel(new VertexLabel(gaussianLabel));
        assertThat(latentGaussianVertex, equalTo(labelGaussianVerted));
        assertThat(latentGaussianVertex.getMu().getValue(0), closeTo(3.0, 1e-10));
        assertThat(labelGaussianVerted.getMu().getValue(2), closeTo(5.0, 1e-10));
        assertThat(latentGaussianVertex.getSigma().getValue().scalar(), closeTo(1.0, 1e-10));
        latentGaussianVertex.sample();
    }

    @Test
    public void loadFailsIfIncorrectVersionUsed() {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Keanu only supports Version 1 of MIR");

        MIR.Model.Builder builder = MIR.Model.newBuilder();
        builder.getPropertiesBuilder().setMirVersionValue(10);
        MIRLoader loader = new MIRLoader();
        loader.loadNetwork(builder.build());
    }

    @Test
    public void loadFailsIfModelHasWrongEntryPoint() {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Keanu only supports loading Keanu generated Graphs");

        MIR.Model.Builder builder = MIR.Model.newBuilder();
        builder.setEntryPointName("Incorrect Entry Point");
        MIRLoader loader = new MIRLoader();
        loader.loadNetwork(builder.build());
    }

    @Test
    public void loadFailsIfEntryPointIsMissing() {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Expected Entry Point not found");

        MIR.Model.Builder builder = MIR.Model.newBuilder();
        builder.setEntryPointName(MIRSaver.ENTRY_POINT_NAME);
        MIRLoader loader = new MIRLoader();
        loader.loadNetwork(builder.build());
    }

    @Test
    public void loadFailsIfEntryPointIsEmpty() {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Entry Point has no Instruction Groups");

        MIR.Model.Builder builder = MIR.Model.newBuilder();
        builder.setEntryPointName(MIRSaver.ENTRY_POINT_NAME);
        MIR.Function.Builder function = MIR.Function.newBuilder();
        function.setName(MIRSaver.ENTRY_POINT_NAME);
        builder.putFunctionsByName(MIRSaver.ENTRY_POINT_NAME, function.build());
        MIRLoader loader = new MIRLoader();
        loader.loadNetwork(builder.build());
    }

    @Test
    public void loadFailsIfTooManyInstructionGroups() {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("More than the expected number of instruction groups");

        MIR.Model.Builder builder = MIR.Model.newBuilder();
        builder.setEntryPointName(MIRSaver.ENTRY_POINT_NAME);
        MIR.Function.Builder function = MIR.Function.newBuilder();
        function.setName(MIRSaver.ENTRY_POINT_NAME);
        function.addInstructionGroupsBuilder().setId(1);
        function.addInstructionGroupsBuilder().setId(2);
        builder.putFunctionsByName(MIRSaver.ENTRY_POINT_NAME, function.build());
        MIRLoader loader = new MIRLoader();
        loader.loadNetwork(builder.build());
    }

    @Test
    public void loadFailsIfNonGraphInstructionGroupPresent() {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Received Non Graph Instruction Group");

        MIR.Model.Builder builder = MIR.Model.newBuilder();
        builder.setEntryPointName(MIRSaver.ENTRY_POINT_NAME);
        MIR.Function.Builder function = MIR.Function.newBuilder();
        function.setName(MIRSaver.ENTRY_POINT_NAME);
        function.addInstructionGroupsBuilder()
            .setId(1)
            .getLoopBuilder();
        builder.putFunctionsByName(MIRSaver.ENTRY_POINT_NAME, function.build());
        MIRLoader loader = new MIRLoader();
        loader.loadNetwork(builder.build());
    }
}
