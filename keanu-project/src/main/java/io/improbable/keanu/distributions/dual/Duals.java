package io.improbable.keanu.distributions.dual;

import java.util.NoSuchElementException;
import java.util.TreeSet;

import com.google.common.collect.Sets;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

public class Duals {
    public static final DualName A = new DualName("A");
    public static final DualName B = new DualName("B");
    public static final DualName K = new DualName("K");
    public static final DualName S = new DualName("S");
    public static final DualName T = new DualName("T");
    public static final DualName X = new DualName("X");
    public static final DualName BETA = new DualName("BETA");
    public static final DualName MU = new DualName("MU");
    public static final DualName SIGMA = new DualName("SIGMA");
    public static final DualName THETA = new DualName("THETA");

    private final TreeSet<Dual> duals = Sets.newTreeSet();

    public Duals put(DualName id, DoubleTensor value) {
        Dual dual = new Dual(id, value);
        if (duals.contains(dual)) {
            throw new IllegalArgumentException("Dual named " + id + " has already been set");
        }
        duals.add(dual);
        return this;
    }

    public Dual get(DualName id) {
        Dual dual = duals.floor(new Dual(id));
        if (dual == null) {
            throw new NoSuchElementException("Cannot find Dual named " + id);
        }
        return dual;
    }
}
