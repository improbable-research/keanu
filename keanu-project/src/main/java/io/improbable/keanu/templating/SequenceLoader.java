package io.improbable.keanu.templating;

import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static io.improbable.keanu.templating.SequenceBuilder.getUnscopedLabel;
import static io.improbable.keanu.templating.SequenceItem.parseSequenceHash;
import static io.improbable.keanu.templating.SequenceItem.parseSequenceItemIndex;
import static io.improbable.keanu.templating.SequenceItem.parseSequenceName;

public class SequenceLoader {

    public static Sequence loadFromBayesNet(BayesianNetwork network) {
        Collection<Sequence> sequences = loadSequences(network, false).values();
        return sequences.stream().findFirst().get();
    }

    public static Map<Integer, Sequence> loadMultipleSequencesFromBayesNet(BayesianNetwork network) {
        return loadSequences(network, true);
    }

    private static Map<Integer, Sequence> loadSequences(BayesianNetwork network, boolean shouldLoadMultiple) {
        List<Vertex> vertices = TopologicalSort.sort(network.getAllVertices());
        Map<Integer, Sequence> sequences = new HashMap<>();
        for (Vertex vertex : vertices) {
            addVertexToSequences(vertex, sequences);
            if (!shouldLoadMultiple && sequences.size() > 1) {
                throw new SequenceConstructionException("The provided BayesianNetwork contains more than one Sequence");
            }
        }
        if (sequences.size() == 0) {
            throw new SequenceConstructionException("The provided BayesianNetwork contains no Sequences");
        }
        return sequences;
    }

    private static void addVertexToSequences(Vertex<?> vertex, Map<Integer, Sequence> sequences) {
        VertexLabel label = vertex.getLabel();
        if (label != null) {
            Optional<Integer> sequenceItemIndex = parseSequenceItemIndex(label);
            if (sequenceItemIndex.isPresent()) {
                Optional<String> sequenceName = parseSequenceName(label);
                int sequenceHash = parseSequenceHash(label, sequenceName.isPresent());

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
