package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorMatchers;
import org.hamcrest.Description;
import org.hamcrest.Matcher;
import org.hamcrest.TypeSafeDiagnosingMatcher;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.nullValue;

public class VertexMatchers {

    public static <T> Matcher<IVertex<T>> hasLabel(VertexLabel label) {
        return hasLabel(equalTo(label));
    }

    public static <T> Matcher<IVertex<T>> hasNoLabel() {
        return hasLabel(nullValue());
    }

    public static <T> Matcher<IVertex<T>> hasLabel(Matcher<? super VertexLabel> labelMatcher) {
        return new TypeSafeDiagnosingMatcher<IVertex<T>>() {
            @Override
            protected boolean matchesSafely(IVertex<T> vertex, Description description) {
                description.appendText("vertex with label ").appendValue(vertex.getLabel());
                return labelMatcher.matches(vertex.getLabel());
            }

            @Override
            public void describeTo(Description description) {
                description.appendText("vertex with label ").appendDescriptionOf(labelMatcher);
            }
        };
    }

    public static <T> Matcher<IVertex<T>> hasParents(Matcher<? super Collection<IVertex<T>>> parentMatcher) {
        return new TypeSafeDiagnosingMatcher<IVertex<T>>() {
            @Override
            protected boolean matchesSafely(IVertex<T> vertex, Description description) {
                description.appendText("vertex with parents ").appendValue(vertex.getParents());
                return parentMatcher.matches(vertex.getParents());
            }

            @Override
            public void describeTo(Description description) {
                description.appendText("vertex with parents ").appendDescriptionOf(parentMatcher);
            }
        };
    }


    public static <DATATYPE, TENSOR extends Tensor<DATATYPE, TENSOR>> Matcher<IVertex<TENSOR>> hasValue(DATATYPE... values) {
        return hasValue(Arrays.stream(values).map(v -> equalTo(v)).collect(Collectors.toList()));
    }

    public static <DATATYPE, TENSOR extends Tensor<DATATYPE, TENSOR>> Matcher<IVertex<TENSOR>> hasValue(List<Matcher<DATATYPE>> valueMatcher) {
        return new TypeSafeDiagnosingMatcher<IVertex<TENSOR>>() {
            @Override
            protected boolean matchesSafely(IVertex<TENSOR> vertex, Description description) {
                description.appendText("vertex with value ").appendValue(vertex.getValue());
                return TensorMatchers.hasValue(valueMatcher).matches(vertex.getValue());
            }

            @Override
            public void describeTo(Description description) {
                description.appendText("vertex with value ").appendValue(valueMatcher);
            }
        };
    }

    public static <DATATYPE, TENSOR extends Tensor<DATATYPE, TENSOR>> Matcher<IVertex<TENSOR>> hasValue(TENSOR tensor) {
        return hasValue(TensorMatchers.valuesAndShapesMatch(tensor));
    }

    public static <T> Matcher<IVertex<T>> hasValue(Matcher<? super T> valueMatcher) {
        return new TypeSafeDiagnosingMatcher<IVertex<T>>() {
            @Override
            protected boolean matchesSafely(IVertex<T> vertex, Description description) {
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
