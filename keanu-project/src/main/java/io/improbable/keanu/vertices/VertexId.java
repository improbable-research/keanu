package io.improbable.keanu.vertices;

import java.util.LinkedList;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

import com.google.common.primitives.Ints;

/**
 * An object representing the ID value of a vertex.  IDs are assigned in such a way that a Lexicographic ordering of
 * all nodes based on their ID value is a valid topological ordering of the graph made up by those vertices.
 *
 * Ids also encapsulate the notion of "Depth".  When we have graphs within graphs, the depth tells us at what level the
 * graph exists - ie depth 1 is the outermost graph, depth 2 is a graph within a graph etc.
 */
public class VertexId implements Comparable<VertexId> {

    public static final AtomicLong ID_GENERATOR = new AtomicLong(0L);

    LinkedList<Long> idValues = new LinkedList<>();

    public VertexId() {
        long newId = ID_GENERATOR.getAndIncrement();
        idValues.push(newId);
    }

    @Override
    public int compareTo(VertexId that) {
        long comparisonValue = 0;

        for (int i = 0; i < Math.min(this.idValues.size(), that.idValues.size()) && comparisonValue == 0; i++) {
            comparisonValue = this.idValues.get(i) - that.idValues.get(i);
        }

        if (comparisonValue == 0) {
            comparisonValue = this.idValues.size() - that.idValues.size();
        }

        return Ints.saturatedCast(comparisonValue);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        VertexId vertexId = (VertexId) o;
        return Objects.equals(idValues, vertexId.idValues);
    }

    @Override
    public int hashCode() {
        return Objects.hash(idValues);
    }

    @Override
    public String toString() {
        return idValues.peek().toString();
    }

    public int getDepth() {
        return idValues.size();
    }
}
