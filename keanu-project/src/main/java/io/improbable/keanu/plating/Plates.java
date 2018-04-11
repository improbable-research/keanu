package io.improbable.keanu.plating;

import java.util.ArrayList;
import java.util.List;

public class Plates {
    private List<Plate> plates;

    public Plates(int reservedSize) {
        plates = new ArrayList<>(reservedSize);
    }

    public int size() {
        return plates.size();
    }

    public void add(Plate p) {
        plates.add(p);
    }

    public List<Plate> asList() {
        return plates;
    }

}
