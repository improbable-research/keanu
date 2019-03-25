package io.improbable.keanu.util;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;

import static io.improbable.keanu.util.DescriptionCreator.createDescriptionAllowingLabels;

public class DescriptionUtils {

    public static String createIfStringDescription(BooleanVertex predicate, Vertex thn, Vertex els, boolean includeBrackets) {
        StringBuilder builder = new StringBuilder();

        if (includeBrackets) {
            builder.append("(");
        }
        builder.append(createDescriptionAllowingLabels(predicate));
        builder.append(" ? ");
        builder.append(createDescriptionAllowingLabels(thn));
        builder.append(" : ");
        builder.append(createDescriptionAllowingLabels(els));
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

        builder.append(createDescriptionAllowingLabels(a));
        builder.append(" ");
        builder.append(operation);
        builder.append(" ");
        builder.append(createDescriptionAllowingLabels(b));

        if (includeBrackets) {
            builder.append(")");
        }
        return builder.toString();
    }
}
