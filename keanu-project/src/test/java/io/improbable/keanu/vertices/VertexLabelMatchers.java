package io.improbable.keanu.vertices;

import org.hamcrest.Description;
import org.hamcrest.Matcher;
import org.hamcrest.TypeSafeDiagnosingMatcher;

import static org.hamcrest.Matchers.equalTo;

public class VertexLabelMatchers {

    public static Matcher<VertexLabel> hasUnqualifiedName(String nameMatcher) {
        return hasUnqualifiedName(equalTo(nameMatcher));

    }

    public static Matcher<VertexLabel> hasUnqualifiedName(Matcher<String> nameMatcher) {
        return new TypeSafeDiagnosingMatcher<VertexLabel>() {
            @Override
            protected boolean matchesSafely(VertexLabel label, Description description) {
                description.appendValue(label);
                return nameMatcher.matches(label.getUnqualifiedName());
            }

            @Override
            public void describeTo(Description description) {
                description.appendText("vertex with unqualified name ").appendDescriptionOf(nameMatcher);
            }
        };
    }
}
