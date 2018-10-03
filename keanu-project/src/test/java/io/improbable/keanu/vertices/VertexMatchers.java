package io.improbable.keanu.vertices;

import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.nullValue;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorMatchers;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import org.hamcrest.Description;
import org.hamcrest.Matcher;
import org.hamcrest.TypeSafeDiagnosingMatcher;

public class VertexMatchers {

  public static <T> Matcher<Vertex<T>> hasLabel(VertexLabel label) {
    return hasLabel(equalTo(label));
  }

  public static <T> Matcher<Vertex<T>> hasNoLabel() {
    return hasLabel(nullValue());
  }

  public static <T> Matcher<Vertex<T>> hasLabel(Matcher<? super VertexLabel> labelMatcher) {
    return new TypeSafeDiagnosingMatcher<Vertex<T>>() {
      @Override
      protected boolean matchesSafely(Vertex<T> vertex, Description description) {
        description.appendText("vertex with label ").appendValue(vertex.getLabel());
        return labelMatcher.matches(vertex.getLabel());
      }

      @Override
      public void describeTo(Description description) {
        description.appendText("vertex with label ").appendDescriptionOf(labelMatcher);
      }
    };
  }

  public static <T> Matcher<Vertex<T>> hasParents(
      Matcher<? super Collection<Vertex<T>>> parentMatcher) {
    return new TypeSafeDiagnosingMatcher<Vertex<T>>() {
      @Override
      protected boolean matchesSafely(Vertex<T> vertex, Description description) {
        description.appendText("vertex with parents ").appendValue(vertex.getParents());
        return parentMatcher.matches(vertex.getParents());
      }

      @Override
      public void describeTo(Description description) {
        description.appendText("vertex with parents ").appendDescriptionOf(parentMatcher);
      }
    };
  }

  public static <DATATYPE, TENSOR extends Tensor<DATATYPE>> Matcher<Vertex<TENSOR>> hasValue(
      DATATYPE... values) {
    return hasValue(Arrays.stream(values).map(v -> equalTo(v)).collect(Collectors.toList()));
  }

  public static <DATATYPE, TENSOR extends Tensor<DATATYPE>> Matcher<Vertex<TENSOR>> hasValue(
      List<Matcher<DATATYPE>> valueMatcher) {
    return new TypeSafeDiagnosingMatcher<Vertex<TENSOR>>() {
      @Override
      protected boolean matchesSafely(Vertex<TENSOR> vertex, Description description) {
        description.appendText("vertex with value ").appendValue(vertex.getValue());
        return TensorMatchers.hasValue(valueMatcher).matches(vertex.getValue());
      }

      @Override
      public void describeTo(Description description) {
        description.appendText("vertex with value ").appendValue(valueMatcher);
      }
    };
  }

  public static <T> Matcher<Vertex<T>> hasValue(Matcher<? super T> valueMatcher) {
    return new TypeSafeDiagnosingMatcher<Vertex<T>>() {
      @Override
      protected boolean matchesSafely(Vertex<T> vertex, Description description) {
        description.appendText("vertex with value ").appendValue(vertex.getValue());
        return valueMatcher.matches(vertex.getValue());
      }

      @Override
      public void describeTo(Description description) {
        description.appendText("vertex with value ").appendDescriptionOf(valueMatcher);
      }
    };
  }
}
