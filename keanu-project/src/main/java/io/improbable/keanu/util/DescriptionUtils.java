package io.improbable.keanu.util;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BooleanBinaryOpVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.BinomialVertex;

import static io.improbable.keanu.util.DescriptionCreator.createDescriptionAllowingLabels;

class DescriptionUtils {

    static String createIfStringDescription(BooleanVertex predicate, Vertex thn, Vertex els, boolean includeBrackets) {
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

    static String createBinomialDescription(BinomialVertex binomialVertex, boolean includeBrackets) {
        String pString = createDescriptionAllowingLabels(binomialVertex.getP());
        String nString = createDescriptionAllowingLabels(binomialVertex.getN());

        return new StringBuilder(includeBrackets ? "(" : "")
            .append("Binomial(p=")
            .append(pString)
            .append(", n=")
            .append(nString)
            .append(")")
            .append(includeBrackets ? ")" : "")
            .toString();
    }

    static <X extends Tensor, Y extends Tensor> String createBooleanBinaryOpDescription(BooleanBinaryOpVertex<X, Y> opVertex, String operation, boolean includeBrackets) {
        StringBuilder builder = new StringBuilder();

        if (includeBrackets) {
            builder.append("(");
        }

        builder.append(createDescriptionAllowingLabels(opVertex.getA()));
        builder.append(operation);
        builder.append(createDescriptionAllowingLabels(opVertex.getB()));

        if (includeBrackets) {
            builder.append(")");
        }
        return builder.toString();
    }
}
