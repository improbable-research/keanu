package io.improbable.keanu.util;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;

public class DescriptionUtils {

    public static String createIfStringDescription(BooleanVertex predicate, Vertex thn, Vertex els, boolean includeBrackets) {
        StringBuilder builder = new StringBuilder();

        if (includeBrackets) {
            builder.append("(");
        }
        builder.append(predicate.createDescriptionAllowingLabels());
        builder.append(" ? ");
        builder.append(thn.createDescriptionAllowingLabels());
        builder.append(" : ");
        builder.append(els.createDescriptionAllowingLabels());
        if (includeBrackets) {
            builder.append(")");
        }

        return builder.toString();
    }

    public static String createBooleanUnaryOpDescription(Vertex a, Vertex b, String operation, boolean includeBrackets) {
        StringBuilder builder = new StringBuilder();

        if (includeBrackets) {
            builder.append("(");
        }

        builder.append(a.createDescriptionAllowingLabels());
        builder.append(" ");
        builder.append(operation);
        builder.append(" ");
        builder.append(b.createDescriptionAllowingLabels());

        if (includeBrackets) {
            builder.append(")");
        }
        return builder.toString();
    }
}
