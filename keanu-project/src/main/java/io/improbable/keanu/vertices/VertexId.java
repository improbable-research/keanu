package io.improbable.keanu.vertices;

import java.util.concurrent.atomic.AtomicLong;

import com.google.common.primitives.Ints;

import lombok.EqualsAndHashCode;

/**
 * An object representing the ID value of a vertex.  IDs are assigned in such a way that a Lexicographic ordering of
 * all nodes based on their ID value is a valid topological ordering of the graph made up by those vertices.
 *
 * Ids also encapsulate the notion of "Depth".  When we have graphs within graphs, the depth tells us at what level the
 * graph exists - ie depth 1 is the outermost graph, depth 2 is a graph within a graph etc.
 */
@EqualsAndHashCode
public class VertexId implements Comparable<VertexId> {

    public static final AtomicLong ID_GENERATOR = new AtomicLong(0L);
    private static final int DEPTH_ONE_ARRAY_SIZE = 1;

    long[] idValues = new long[DEPTH_ONE_ARRAY_SIZE];

    public VertexId() {
        idValues[0] = ID_GENERATOR.getAndIncrement();
    }

    public void increaseDepth(VertexId prefix) {
        long[] newIdValues = new long[idValues.length + prefix.idValues.length];
        System.arraycopy(prefix.idValues, 0, newIdValues, 0, prefix.idValues.length);
        System.arraycopy(idValues, 0, newIdValues, prefix.idValues.length, idValues.length);
        idValues = newIdValues;
    }

    public void resetID() {
        idValues = new long[DEPTH_ONE_ARRAY_SIZE];
        idValues[0] = ID_GENERATOR.getAndIncrement();
    }

    @Override
    public int compareTo(VertexId that) {
        long comparisonValue = 0;
        int minDepth = Math.min(this.idValues.length, that.idValues.length);

        for (int i = 0; i < minDepth && comparisonValue == 0; i++) {
            comparisonValue = this.idValues[i] - that.idValues[i];
        }

        if (comparisonValue == 0) {
            comparisonValue = this.idValues.length - that.idValues.length;
        }

        return Ints.saturatedCast(comparisonValue);
    }

    @Override
    public String toString() {
        return idValues.toString();
    }

    public int getDepth() {
        return idValues.length;
    }
}
