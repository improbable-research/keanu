package io.improbable.keanu.vertices;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public interface Probabilistic<T> extends Observable<T> {

  /**
   * This is the natural log of the probability at the supplied value. In the case of continuous
   * vertices, this is actually the log of the density, which will differ from the probability by a
   * constant.
   *
   * @param value The supplied value.
   * @return The natural log of the probability function at the supplied value. For continuous
   *     variables this is called the PDF (probability density function). For discrete variables
   *     this is called the PMF (probability mass function).
   */
  double logProb(T value);

  /**
   * The partial derivatives of the natural log prob.
   *
   * @param atValue at a given value
   * @param withRespectTo list of parents to differentiate with respect to
   * @return the partial derivatives of the log of the probability function at the supplied value.
   *     For continuous variables this is called the PDF (probability density function). For
   *     discrete variables this is called the PMF (probability mass function).
   */
  Map<Vertex, DoubleTensor> dLogProb(T atValue, Set<? extends Vertex> withRespectTo);

  default Map<Vertex, DoubleTensor> dLogProb(T atValue, Vertex... withRespectTo) {
    return dLogProb(atValue, new HashSet<>(Arrays.asList(withRespectTo)));
  }

  T getValue();

  void setValue(T value);

  default double logProbAtValue() {
    return logProb(getValue());
  }

  default Map<Vertex, DoubleTensor> dLogProbAtValue(Set<? extends Vertex> withRespectTo) {
    return dLogProb(getValue(), withRespectTo);
  }

  default Map<Vertex, DoubleTensor> dLogProbAtValue(Vertex... withRespectTo) {
    return dLogProb(getValue(), withRespectTo);
  }

  static <V extends Vertex & Probabilistic> List<V> keepOnlyProbabilisticVertices(
      Iterable<? extends Vertex> vertices) {
    ImmutableList.Builder<V> probabilisticVertices = ImmutableList.builder();
    for (Vertex v : vertices) {
      if (v instanceof Probabilistic) {
        probabilisticVertices.add((V) v);
      }
    }
    return probabilisticVertices.build();
  }
}
