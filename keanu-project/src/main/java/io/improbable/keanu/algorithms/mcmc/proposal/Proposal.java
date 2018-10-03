package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.vertices.Vertex;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class Proposal {

  private final Map<Vertex, Object> perVertexProposalTo;
  private final Map<Vertex, Object> perVertexProposalFrom;

  public Proposal() {
    this.perVertexProposalTo = new HashMap<>();
    this.perVertexProposalFrom = new HashMap<>();
  }

  public <T> void setProposal(Vertex<T> vertex, T to) {
    perVertexProposalFrom.put(vertex, vertex.getValue());
    perVertexProposalTo.put(vertex, to);
  }

  public <T> T getProposalTo(Vertex<T> vertex) {
    return (T) perVertexProposalTo.get(vertex);
  }

  public <T> T getProposalFrom(Vertex<T> vertex) {
    return (T) perVertexProposalFrom.get(vertex);
  }

  public Set<Vertex> getVerticesWithProposal() {
    return perVertexProposalTo.keySet();
  }

  public void apply() {
    Set<Vertex> vertices = perVertexProposalTo.keySet();
    for (Vertex v : vertices) {
      v.setValue(getProposalTo(v));
    }
  }

  public void reject() {
    Set<Vertex> vertices = perVertexProposalTo.keySet();
    for (Vertex v : vertices) {
      v.setValue(getProposalFrom(v));
    }
  }
}
