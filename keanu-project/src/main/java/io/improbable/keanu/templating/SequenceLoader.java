package io.improbable.keanu.templating;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static io.improbable.keanu.templating.SequenceBuilder.getUnscopedLabel;
import static io.improbable.keanu.templating.SequenceItem.getSequenceHash;
import static io.improbable.keanu.templating.SequenceItem.getSequenceItemIndex;
import static io.improbable.keanu.templating.SequenceItem.getSequenceName;

public class SequenceLoader {

    public static Sequence loadFromBayesNet(BayesianNetwork network) {
        Collection<Sequence> sequences = loadMultipleSequencesFromBayesNet(network);
        if (sequences.size() != 1) {
            throw new SequenceConstructionException("The provided BayesianNetwork contains more than one Sequence");
        }
        return sequences.stream().findFirst().get();
    }

    public static Collection<Sequence> loadMultipleSequencesFromBayesNet(BayesianNetwork network) {
        List<Vertex> vertices = network.getAllVertices();
        Map<Integer, Sequence> sequences = new HashMap<>();
        for (Vertex vertex : vertices) {
            addVertexToSequences(vertex, sequences);
        }
        return sequences.values();
    }

    private static void addVertexToSequences(Vertex<?> vertex, Map<Integer, Sequence> sequences) {
        VertexLabel label = vertex.getLabel();
        if (label != null) {
            Optional<Integer> sequenceItemIndex = getSequenceItemIndex(label);
            if (sequenceItemIndex.isPresent()) {
                Optional<String> sequenceName = getSequenceName(label);
                int sequenceHash = getSequenceHash(label, sequenceName.isPresent());

                Sequence sequence = getOrCreateSequence(sequences, sequenceHash, sequenceName.orElse(null));
                SequenceItem item = getOrCreateSequenceItem(sequence, sequenceItemIndex.get(), sequenceHash, sequenceName.orElse(null));

                //Removes the scope from a label because this is required by the sequenceItem.add() method
                VertexLabel newLabel = getUnscopedLabel(label, sequenceName.isPresent());
                vertex.setLabel(newLabel);

                item.add(vertex);
            }
        }
    }

    private static boolean sequenceContainsKey(Sequence sequence, int index) {
        List<SequenceItem> sequenceItems = sequence.asList();
        if (index >= sequenceItems.size()) {
            return false;
        }
        return sequenceItems.get(index) != null;
    }

    private static Sequence getOrCreateSequence(Map<Integer, Sequence> sequences, int sequenceHash, String sequenceName) {
        Sequence sequence;
        if (sequences.containsKey(sequenceHash)) {
            sequence = sequences.get(sequenceHash);
        } else {
            sequence = new Sequence(0, sequenceHash, sequenceName);
            sequences.put(sequenceHash, sequence);
        }
        return sequence;
    }

    private static SequenceItem getOrCreateSequenceItem(Sequence sequence, Integer sequenceItemIndex, int sequenceHash, String sequenceName) {
        SequenceItem sequenceItem;
        if (sequenceContainsKey(sequence, sequenceItemIndex)) {
            sequenceItem = sequence.asList().get(sequenceItemIndex);
        } else {
            sequenceItem = new SequenceItem(sequenceItemIndex, sequenceHash, sequenceName);
            sequence.add(sequenceItem);
        }
        return sequenceItem;
    }
}
