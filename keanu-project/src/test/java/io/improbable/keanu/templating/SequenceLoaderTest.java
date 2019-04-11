package io.improbable.keanu.templating;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.io.ProtobufLoader;
import io.improbable.keanu.util.io.ProtobufSaver;
import io.improbable.keanu.vertices.SimpleVertexDictionary;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexDictionary;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.function.Consumer;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;

public class SequenceLoaderTest {

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    @Test
    public void youCanConstructSingleSequenceItem() throws IOException {
        VertexLabel xLabel = new VertexLabel("x");

        DoubleVertex two = new ConstantDoubleVertex(2.0);

        Consumer<SequenceItem> factory = sequenceItem -> {
            DoubleProxyVertex xInput = sequenceItem.addDoubleProxyFor(xLabel);
            DoubleVertex xOutput = xInput.multiply(two).setLabel(xLabel);

            sequenceItem.add(xOutput);
        };

        DoubleVertex xInitial = new ConstantDoubleVertex(1.0).setLabel(xLabel);
        VertexDictionary initialState = SimpleVertexDictionary.of(xInitial);

        Sequence sequence = new SequenceBuilder()
            .withInitialState(initialState)
            .count(2)
            .withFactory(factory)
            .build();
        BayesianNetwork network = sequence.toBayesianNetwork();

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        ProtobufSaver saver = new ProtobufSaver(network);
        saver.save(writer, true);
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork reloadedNetwork = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
        Sequence reconstructedSequence = SequenceLoader.loadFromBayesNet(reloadedNetwork);

        assertThat(reconstructedSequence.size(), is(2));
        assertThat(reconstructedSequence.getUniqueIdentifier(), is(sequence.getUniqueIdentifier()));

        List<SequenceItem> originalList = sequence.asList();
        VertexLabel xProxyLabel = SequenceBuilder.proxyLabelFor(xLabel);
        reconstructedSequence.forEach(sequenceItem -> {
            SequenceItem originalItem = originalList.get(sequenceItem.getIndex());
            assertThat(sequenceItem.getContents().keySet(), is(originalItem.getContents().keySet()));
            assertThat(sequenceItem.get(xLabel), notNullValue());
            assertThat(sequenceItem.get(xProxyLabel), notNullValue());
        });

        Vertex<? extends DoubleTensor> outputVertex = reconstructedSequence.getLastItem().get(xLabel);
        double actualOutputValue = outputVertex.getValue().scalar();
        assertThat(actualOutputValue, is(4.0));
    }

    @Test
    public void youCanConstructASequenceItem() throws IOException {
        VertexLabel xLabel = new VertexLabel("x");
        Sequence sequence = constructSimpleSequence(null, xLabel);
        BayesianNetwork network = sequence.toBayesianNetwork();

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        ProtobufSaver saver = new ProtobufSaver(network);
        saver.save(writer, true);
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork reconstructedNetwork = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
        Sequence reconstructedSequence = SequenceLoader.loadFromBayesNet(reconstructedNetwork);

        assertThat(reconstructedSequence.size(), is(2));
        assertThat(reconstructedSequence.getUniqueIdentifier(), is(sequence.getUniqueIdentifier()));

        assertSequenceContains(sequence, reconstructedSequence, xLabel, SequenceBuilder.proxyLabelFor(xLabel));
    }

    @Test
    public void itThrowsWhenManyAreStoredButOneIsRequested() throws IOException {
        expectedException.expect(SequenceConstructionException.class);
        expectedException.expectMessage("The provided BayesianNetwork contains more than one Sequence");

        VertexLabel x1Label = new VertexLabel("x1");
        VertexLabel x2Label = new VertexLabel("x2");
        VertexLabel outputLabel = new VertexLabel("OUTPUT");
        String sequence1Label = "Sequence_1";
        String sequence2Label = "Sequence_2";

        Sequence sequence1 = constructSimpleSequence(sequence1Label, x1Label);
        Sequence sequence2 = constructSimpleSequence(sequence2Label, x2Label);

        DoubleVertex output1 = sequence1.getLastItem().get(x1Label);
        DoubleVertex output2 = sequence2.getLastItem().get(x2Label);

        DoubleVertex masterOutput = output1.multiply(output2).setLabel(outputLabel);

        BayesianNetwork network = new BayesianNetwork(masterOutput.getConnectedGraph());

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        ProtobufSaver saver = new ProtobufSaver(network);
        saver.save(writer, true);
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork reconstructedNetwork = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
        Sequence reconstructedSequence = SequenceLoader.loadFromBayesNet(reconstructedNetwork);
    }

    @Test
    public void youCanConstructManySequenceItems() throws IOException {
        VertexLabel x1Label = new VertexLabel("x1");
        VertexLabel x2Label = new VertexLabel("x2");
        VertexLabel outputLabel = new VertexLabel("OUTPUT");
        String sequence1Label = "Sequence_1";
        String sequence2Label = "Sequence_2";

        Sequence sequence1 = constructSimpleSequence(sequence1Label, x1Label);
        Sequence sequence2 = constructSimpleSequence(sequence2Label, x2Label);

        DoubleVertex output1 = sequence1.getLastItem().get(x1Label);
        DoubleVertex output2 = sequence2.getLastItem().get(x2Label);

        DoubleVertex masterOutput = output1.multiply(output2).setLabel(outputLabel);

        BayesianNetwork network = new BayesianNetwork(masterOutput.getConnectedGraph());

        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        ProtobufSaver saver = new ProtobufSaver(network);
        saver.save(writer, true);
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork reconstructedNetwork = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
        Collection<Sequence> reconstructedSequences = SequenceLoader.loadMultipleSequencesFromBayesNet(reconstructedNetwork).values();

        Sequence reconstructedSequence1 = getSequenceAndCheckNotNull(reconstructedSequences, sequence1Label);

        Sequence reconstructedSequence2 = getSequenceAndCheckNotNull(reconstructedSequences, sequence2Label);

        assertThat(reconstructedSequences.size(), is(2));
        assertThat(reconstructedSequence1.getUniqueIdentifier(), is(sequence1.getUniqueIdentifier()));
        assertThat(reconstructedSequence2.getUniqueIdentifier(), is(sequence2.getUniqueIdentifier()));
        assertThat(reconstructedSequence1.size(), is(2));
        assertThat(reconstructedSequence2.size(), is(2));
        assertThat(reconstructedSequence1.getName(), is(sequence1Label));
        assertThat(reconstructedSequence2.getName(), is(sequence2Label));

        VertexLabel x1ProxyLabel = SequenceBuilder.proxyLabelFor(x1Label);
        VertexLabel x2ProxyLabel = SequenceBuilder.proxyLabelFor(x2Label);
        assertSequenceContains(sequence1, reconstructedSequence1, x1Label, x1ProxyLabel);
        assertSequenceContains(sequence2, reconstructedSequence2, x2Label, x2ProxyLabel);

        BayesianNetwork reconstructedNetwork1 = sequence1.toBayesianNetwork();
        Vertex reconstructedMasterOutput = reconstructedNetwork1.getVertexByLabel(outputLabel);
        assertThat(reconstructedMasterOutput, notNullValue());
        assertThat(((DoubleTensor) reconstructedMasterOutput.getValue()).scalar(), is(16.0));
    }

    private void assertSequenceContains(Sequence sequence, Sequence reconstructedSequence, VertexLabel xLabel, VertexLabel xProxyLabel) {
        List<SequenceItem> originalList = sequence.asList();
        reconstructedSequence.forEach(sequenceItem -> {
            SequenceItem originalItem = originalList.get(sequenceItem.getIndex());
            assertThat(sequenceItem.getContents().keySet(), is(originalItem.getContents().keySet()));
            assertThat(sequenceItem.get(xLabel), notNullValue());
            assertThat(sequenceItem.get(xProxyLabel), notNullValue());
        });

        Vertex<? extends DoubleTensor> outputVertex2 = reconstructedSequence.getLastItem().get(xLabel);
        double actualOutputValue2 = outputVertex2.getValue().scalar();
        assertThat(actualOutputValue2, is(4.0));
    }

    private Sequence getSequenceAndCheckNotNull(Collection<Sequence> sequences, String sequenceName) {
        Sequence reconstructedSequence = sequences
            .stream()
            .filter(sequence -> sequence.getName().equals(sequenceName))
            .findFirst()
            .orElse(null);
        assertThat(reconstructedSequence, notNullValue());
        return reconstructedSequence;
    }

    private Sequence constructSimpleSequence(String sequenceName, VertexLabel outputLabel) {
        DoubleVertex two = new ConstantDoubleVertex(2.0);

        Consumer<SequenceItem> factory = sequenceItem -> {
            DoubleProxyVertex xInput = sequenceItem.addDoubleProxyFor(outputLabel);
            DoubleVertex xOutput = xInput.multiply(two).setLabel(outputLabel);

            sequenceItem.add(xOutput);
        };

        DoubleVertex xInitial = new ConstantDoubleVertex(1.0).setLabel(outputLabel);
        VertexDictionary initialState = SimpleVertexDictionary.of(xInitial);

        SequenceBuilder builder = new SequenceBuilder();
        if (sequenceName != null) {
            builder = builder.named(sequenceName);
        }

        return builder
            .withInitialState(initialState)
            .count(2)
            .withFactory(factory)
            .build();
    }

    @Test
    public void failsIfThereAreNoSequences() throws IOException {
        expectedException.expect(SequenceConstructionException.class);
        expectedException.expectMessage("The provided BayesianNetwork contains no Sequences");

        IntegerVertex two = new ConstantIntegerVertex(1).plus(new ConstantIntegerVertex(1));
        BayesianNetwork network = new BayesianNetwork(two.getConnectedGraph());
        ByteArrayOutputStream writer = new ByteArrayOutputStream();
        ProtobufSaver saver = new ProtobufSaver(network);
        saver.save(writer, true);
        ProtobufLoader loader = new ProtobufLoader();
        BayesianNetwork reconstructedNetwork = loader.loadNetwork(new ByteArrayInputStream(writer.toByteArray()));
        Sequence reconstructedSeqeuence = SequenceLoader.loadFromBayesNet(reconstructedNetwork);
    }

}
