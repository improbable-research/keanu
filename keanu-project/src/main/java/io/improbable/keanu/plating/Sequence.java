package io.improbable.keanu.plating;

import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class Sequence implements Iterable<SequenceItem> {

    private List<SequenceItem> containedItems;

    public Sequence(int reservedSize) {
        containedItems = new ArrayList<>(reservedSize);
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
}
