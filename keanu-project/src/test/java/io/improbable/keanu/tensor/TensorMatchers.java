package io.improbable.keanu.tensor;

import org.hamcrest.Description;
import org.hamcrest.Matcher;
import org.hamcrest.TypeSafeDiagnosingMatcher;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.hamcrest.Matchers.both;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.equalTo;


public class TensorMatchers {
    private TensorMatchers() {
    }

    public static <T> Matcher<Tensor<T>> hasShape(long... shape) {
        return hasShape(equalTo(shape));
    }

    public static <T> Matcher<Tensor<T>> hasShape(Matcher<long[]> shapeMatcher) {
        return new TypeSafeDiagnosingMatcher<Tensor<T>>() {
            @Override
            protected boolean matchesSafely(Tensor<T> item, Description mismatchDescription) {
                mismatchDescription.appendText("had shape ").appendValue(item.getShape());
                return shapeMatcher.matches(item.getShape());
            }

            @Override
            public void describeTo(Description description) {
                description.appendText("Tensor with shape ").appendValue(shapeMatcher);
            }
        };
    }

    public static <T> Matcher<Tensor<T>> isScalarWithValue(T value) {
        return isScalarWithValue(equalTo(value));
    }

    public static <T> Matcher<Tensor<T>> isScalarWithValue(Matcher<T> value) {
        return new TypeSafeDiagnosingMatcher<Tensor<T>>() {
            @Override
            protected boolean matchesSafely(Tensor<T> item, Description mismatchDescription) {
                mismatchDescription.appendValue(item);
                return item.isScalar() && value.matches(item.getValue(0));
            }

            @Override
            public void describeTo(Description description) {
                description.appendText("Scalar with value ").appendValue(value);
            }
        };
    }

    public static <T> Matcher<Tensor<T>> hasValue(T... values) {
        return hasValue(Arrays.stream(values).map(v -> equalTo(v)).collect(Collectors.toList()));
    }

    public static <T> Matcher<Tensor<T>> hasValue(List<Matcher<T>> valueMatchers) {
        return new TypeSafeDiagnosingMatcher<Tensor<T>>() {
            @Override
            protected boolean matchesSafely(Tensor<T> item, Description mismatchDescription) {
                mismatchDescription.appendText("Tensor");
                T[] itemArray = item.asFlatArray();
                if (itemArray.length != valueMatchers.size()) {
                    mismatchDescription
                        .appendText(" with different size ")
                        .appendValue(itemArray.length);
                    return false;
                }
                for (int i = 0; i < valueMatchers.size(); i++) {
                    if (!valueMatchers.get(i).matches(itemArray[i])) {
                        mismatchDescription
                            .appendText(" with different value ")
                            .appendValue(itemArray[i])
                            .appendText(" at entry ")
                            .appendValue(i);
                        return false;
                    }
                }
                return true;
            }

            @Override
            public void describeTo(Description description) {
                description.appendText("Tensor with value ").appendValue(valueMatchers);
            }
        };
    }

    public static <T> Matcher<Tensor<T>> valuesAndShapesMatch(Tensor<T> tensor) {
        return both(TensorMatchers.valuesMatch(tensor)).and(hasShape(tensor.getShape()));
    }

    public static <T> Matcher<Tensor<T>> allValues(Matcher<T> valueMatcher) {
        return new TypeSafeDiagnosingMatcher<Tensor<T>>() {
            @Override
            protected boolean matchesSafely(Tensor<T> item, Description mismatchDescription) {
                mismatchDescription.appendText("Tensor");
                Tensor.FlattenedView<T> itemFlattened = item.getFlattenedView();

                for (int i = 0; i < itemFlattened.size(); i++) {
                    if (!valueMatcher.matches(itemFlattened.getOrScalar(i))) {
                        mismatchDescription
                            .appendText(" with different value ")
                            .appendValue(itemFlattened.getOrScalar(i))
                            .appendText(" at entry ")
                            .appendValue(i);
                        return false;
                    }
                }
                return true;
            }

            @Override
            public void describeTo(Description description) {
                description.appendText("Tensor with all values ").appendValue(valueMatcher);
            }
        };
    }

    public static Matcher<NumberTensor> lessThanOrEqualTo(NumberTensor other) {
        return new TypeSafeDiagnosingMatcher<NumberTensor>() {
            @Override
            protected boolean matchesSafely(NumberTensor item, Description mismatchDescription) {
                mismatchDescription.appendText("Tensor ").appendValue(item);
                return item.lessThanOrEqual(other).allTrue();
            }

            @Override
            public void describeTo(Description description) {
                description.appendText("Tensor ").appendValue(other);
            }
        };
    }

    public static <T> Matcher<Tensor<T>> valuesMatch(Tensor<T> other) {
        return hasValue(other.asFlatArray());
    }

    public static Matcher<Tensor<Double>> allCloseTo(double epsilon, Tensor<Double> other) {
        return hasValue(other.asFlatList().stream().map(v -> closeTo(v, epsilon)).collect(Collectors.toList()));
    }
}
