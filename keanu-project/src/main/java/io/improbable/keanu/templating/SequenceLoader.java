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
import static java.lang.Integer.parseInt;

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
                String sequenceName = getSequenceName(label);
                int sequenceHash = getSequenceHash(label, sequenceName);

                Sequence sequence = getOrCreateSequence(sequences, sequenceHash, sequenceName);
                SequenceItem item = getOrCreateSequenceItem(sequence, sequenceItemIndex.get(), sequenceHash, sequenceName);

                //Removes the scope from a label because this is required by the sequenceItem.add() method
                VertexLabel newLabel = getUnscopedLabel(label, sequenceName);
                vertex.setLabel(newLabel);

                item.add(vertex);
            }
        }
    }

    /**
     * Finds if a vertex is part of a SequenceItem or not.
     * @param label label of the vertex being parsed.
     * @return -1 if not in sequenceItem. Otherwise returns sequenceItem index.
     */
    private static Optional<Integer> getSequenceItemIndex(VertexLabel label) {
        String outerNamespace = label.getOuterNamespace().orElse(null);
        if (outerNamespace == null) {
            return Optional.empty();
        }
        if (outerNamespace.startsWith(SequenceItem.NAME_PREFIX)) {
            return Optional.of(parseInt(outerNamespace.replaceFirst(SequenceItem.NAME_PREFIX, "")));
        }
        outerNamespace = label.withoutOuterNamespace().getOuterNamespace().orElse(null);
        if (outerNamespace == null) {
            return Optional.empty();
        }
        return Optional.of(parseInt(outerNamespace.replaceFirst(SequenceItem.NAME_PREFIX, "")));
    }

    /**
     * Tries to get the unique sequence name from a vertex label
     * @param label
     * @return will return null if there is not a unique sequence name
     */
    private static String getSequenceName(VertexLabel label) {
        String outerNamespace = label.getOuterNamespace().orElse(null);
        if (outerNamespace != null && outerNamespace.startsWith(SequenceItem.NAME_PREFIX)) {
            outerNamespace = null;
        }
        return outerNamespace;
    }

    private static int getSequenceHash(VertexLabel label, String sequenceName) {
        VertexLabel withHashAsOuterNamespace = label.withoutOuterNamespace();
        if (sequenceName != null) {
            withHashAsOuterNamespace = withHashAsOuterNamespace.withoutOuterNamespace();
        }
        String hashLabel = withHashAsOuterNamespace.getOuterNamespace()
            .orElseThrow(() -> new SequenceConstructionException("Could not parse the sequence hash in the vertex label"));
        return parseInt(hashLabel);
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
