package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/** Saves the state (value and observed) of a specified collection of vertices. */
public class NetworkSnapshot {

  public static NetworkSnapshot create(Set<Vertex> vertices) {
    return new NetworkSnapshot(vertices);
  }

  private final Map<Vertex, Object> values;
  private final Set<Vertex> observed;

  private NetworkSnapshot(Collection<Vertex> vertices) {
    values = new HashMap<>();
    observed = new HashSet<>();
    for (Vertex v : vertices) {
      values.put(v, v.getValue());
      if (v.isObserved()) {
        observed.add(v);
      }
    }
  }

  /** Revert the state of the network to the previously saved values */
  public void apply() {
    for (Vertex v : values.keySet()) {
      if (observed.contains(v)) {
        v.observe(values.get(v));
      } else {
        v.unobserve();
        v.setValue(values.get(v));
      }
    }
  }
}
