package io.improbable.keanu.plating;

import java.util.ArrayList;
import java.util.List;

public class Plates {
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

}
