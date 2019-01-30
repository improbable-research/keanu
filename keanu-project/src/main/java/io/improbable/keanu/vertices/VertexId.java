package io.improbable.keanu.vertices;

import com.google.common.primitives.Ints;
import io.improbable.keanu.algorithms.VariableReference;
import lombok.EqualsAndHashCode;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicLong;

/**
 * An object representing the ID value of a vertex.  IDs are assigned in such a way that a Lexicographic ordering of
 * all nodes based on their ID value is a valid topological ordering of the graph made up by those vertices.
 *
 * Ids also encapsulate the notion of "Depth".  When we have graphs within graphs, the depth tells us at what level the
 * graph exists - ie depth 1 is the outermost graph, depth 2 is a graph within a graph etc.
 */
@EqualsAndHashCode
public class VertexId implements Comparable<VertexId>, VariableReference {

    public static final AtomicLong ID_GENERATOR = new AtomicLong(0L);
    private static final int TOP_LEVEL_ARRAY_SIZE = 1;

    private long[] idValues = new long[TOP_LEVEL_ARRAY_SIZE];

    public VertexId() {
        idValues[0] = ID_GENERATOR.getAndIncrement();
    }

    public void addPrefix(VertexId prefix) {
        long[] newIdValues = new long[idValues.length + prefix.idValues.length];
        System.arraycopy(prefix.idValues, 0, newIdValues, 0, prefix.idValues.length);
        System.arraycopy(idValues, 0, newIdValues, prefix.idValues.length, idValues.length);
        idValues = newIdValues;
    }

    public void resetID() {
        idValues = new long[TOP_LEVEL_ARRAY_SIZE];
        idValues[0] = ID_GENERATOR.getAndIncrement();
    }

    public VertexId(long id) {
        idValues[0] = id;
    }

    @Override
    public int compareTo(VertexId that) {
        long comparisonValue = 0;
        int minDepth = Math.min(this.idValues.length, that.idValues.length);

        for (int i = 0; i < minDepth && comparisonValue == 0; i++) {
            comparisonValue = this.idValues[i] - that.idValues[i];
        }

        if (comparisonValue == 0) {
            comparisonValue = (long) this.idValues.length - that.idValues.length;
        }

        return Ints.saturatedCast(comparisonValue);
    }

    public boolean prefixMatches(VertexId prefix) {
        if (prefix.idValues.length > idValues.length) {
            return false;
        }

        for (int i = 0; i < prefix.idValues.length; i++) {
            if (idValues[i] != prefix.idValues[i]) {
                return false;
            }
        }

        return true;
    }

    @Override
    public String toString() {
        return Arrays.toString(idValues);
    }

    public int getIndentation() {
        return idValues.length;
    }

    public long[] getValue() {
        return Arrays.copyOf(idValues, idValues.length);
    }
}
