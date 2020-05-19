package io.improbable.keanu.templating;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;

public class Sequence implements Iterable<SequenceItem> {

    private final ArrayList<SequenceItem> containedItems;
    private final int uniqueIdentifier;
    private final String name;

    public Sequence(int reservedSize, int uniqueIdentifier, String name) {
        containedItems = new ArrayList<>(reservedSize);
        this.uniqueIdentifier = uniqueIdentifier;
        this.name = name;
    }

    public int getUniqueIdentifier() {
        return this.uniqueIdentifier;
    }

    public String getName() {
        return this.name;
    }

    public int size() {
        return containedItems.size();
    }

    public void add(SequenceItem p) {
        containedItems.add(p);
    }

    public List<SequenceItem> asList() {
        return containedItems;
    }

    @NotNull
    @Override
    public Iterator<SequenceItem> iterator() {
        return containedItems.iterator();
    }

    public SequenceItem getLastItem() {
        if (containedItems.isEmpty()) {
            throw new SequenceConstructionException("Sequence is empty!");
        }
        return this.asList().get(this.size() - 1);
    }

    public BayesianNetwork toBayesianNetwork() {
        if (containedItems.isEmpty()) {
            throw new RuntimeException("Bayesian Network construction failed because the Sequence contains no SequenceItems");
        }
        Optional<Vertex<?, ?>> seedVertex = containedItems
            .get(0)
            .getContents()
            .values()
            .stream()
            .findFirst();
        if (!seedVertex.isPresent()) {
            throw new RuntimeException("Bayesian Network construction failed because there are no vertices in the Sequence");
        }
        return new BayesianNetwork(seedVertex.get().getConnectedGraph());
    }
}
