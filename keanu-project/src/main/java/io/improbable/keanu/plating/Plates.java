package io.improbable.keanu.plating;

import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class Plates implements Iterable<Plate> {

    private List<Plate> containedPlates;

    public Plates(int reservedSize) {
        containedPlates = new ArrayList<>(reservedSize);
    }

    public int size() {
        return containedPlates.size();
    }

    public void add(Plate p) {
        containedPlates.add(p);
    }

    public List<Plate> asList() {
        return containedPlates;
    }

    @NotNull
    @Override
    public Iterator<Plate> iterator() {
        return containedPlates.iterator();
    }

    public Plate getLastPlate() {
        if (containedPlates.isEmpty()) {
            throw new PlateConstructionException("Plates is empty!");
        }
        return this.asList().get(this.size() - 1);
    }
}
