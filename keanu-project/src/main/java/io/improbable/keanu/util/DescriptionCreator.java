package io.improbable.keanu.util;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.BooleanIfVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.AndBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BooleanBinaryOpVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.OrBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.EqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanOrEqualVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanOrEqualVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NotEqualsVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleIfVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DifferenceVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DivisionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.IntegerIfVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerAdditionVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerDifferenceVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMultiplicationVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.BinomialVertex;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

public class DescriptionCreator {

    private static Map<Class, String> delimiters = new HashMap<>();
    private static Map<Class, String> infixes = new HashMap<>();

    static {
        //TODO: put in delimiters for class
        delimiters.put(AdditionVertex.class, " + ");
        delimiters.put(IntegerAdditionVertex.class, " + ");
        delimiters.put(DifferenceVertex.class, " - ");
        delimiters.put(IntegerDifferenceVertex.class, " - ");
        delimiters.put(MultiplicationVertex.class, " * ");
        delimiters.put(IntegerMultiplicationVertex.class, " * ");
        delimiters.put(DivisionVertex.class, " / ");
        infixes.put(AndBinaryVertex.class, "&&");
        infixes.put(OrBinaryVertex.class, "||");
        infixes.put(EqualsVertex.class, "==");
        infixes.put(GreaterThanOrEqualVertex.class, ">=");
        infixes.put(GreaterThanVertex.class, ">");
        infixes.put(LessThanOrEqualVertex.class, "<=");
        infixes.put(LessThanVertex.class, "<");
        infixes.put(NotEqualsVertex.class, "!=");
    }

    private static <T extends Tensor> String getBaseDescription(Vertex<T> vertex) {
        if (vertex instanceof ConstantVertex) {
            Tensor value = vertex.getValue();
            String valueString = value.isScalar() ? value.scalar().toString() :
                new StringBuilder("(Tensor with shape: ").append(Arrays.toString(value.getShape())).append(")").toString();

            return new StringBuilder()
                .append("Const(")
                .append(valueString)
                .append(")")
                .toString();
        } else {
            return "<<Unlabelled leaf Vertex>>";
        }
    }

    static <T extends Tensor> String createDescriptionAllowingLabels(Vertex<T> vertex) {
        Collection<Vertex> parents = vertex.getParents();

        if (vertex.getLabel() != null) {
            return vertex.getLabel().toString();
        }

        if (parents.size() == 0) {
            return getBaseDescription(vertex);
        }

        return recursiveDescriptionStep(vertex, true);
    }

    private static <T extends Tensor> String recursiveDescriptionStep(Vertex<T> vertex, boolean includeBrackets) {
        if (vertex instanceof BinomialVertex) {
            String pString = createDescriptionAllowingLabels(((BinomialVertex) vertex).getP());
            String nString = createDescriptionAllowingLabels(((BinomialVertex) vertex).getN());

            return new StringBuilder(includeBrackets ? "(" : "")
                .append("Binomial(p=")
                .append(pString)
                .append(", n=")
                .append(nString)
                .append(")")
                .append(includeBrackets ? ")" : "")
                .toString();
        } else if (vertex instanceof BooleanIfVertex) {
            return DescriptionUtils.createIfStringDescription(
                ((BooleanIfVertex) vertex).getPredicate(),
                ((BooleanIfVertex) vertex).getThn(),
                ((BooleanIfVertex) vertex).getEls(),
                includeBrackets);
        } else if (vertex instanceof BooleanBinaryOpVertex) {
            return DescriptionUtils.createBooleanUnaryOpDescription(
                ((BooleanBinaryOpVertex) vertex).getA(),
                ((BooleanBinaryOpVertex) vertex).getB(),
                getInfixSymbol(vertex),
                includeBrackets);
        } else if (vertex instanceof DoubleIfVertex) {
            return DescriptionUtils.createIfStringDescription(
                ((DoubleIfVertex) vertex).getPredicate(),
                ((DoubleIfVertex) vertex).getThn(),
                ((DoubleIfVertex) vertex).getEls(),
                includeBrackets);
        } else if (vertex instanceof IntegerIfVertex) {
            return DescriptionUtils.createIfStringDescription(
                ((IntegerIfVertex) vertex).getPredicate(),
                ((IntegerIfVertex) vertex).getThn(),
                ((IntegerIfVertex) vertex).getEls(),
                includeBrackets);
        }

        Stream<String> parentStream = vertex
            .getParents()
            .stream()
            .map(DescriptionCreator::createDescriptionAllowingLabels);

        String[] parentStrings = parentStream.toArray(String[]::new);

        CharSequence delimiter = getDescriptionDelimiter(vertex);

        StringBuilder builder = new StringBuilder();

        if (includeBrackets) {
            builder.append("(");
        }
        builder.append(String.join(delimiter, parentStrings));
        if (includeBrackets) {
            builder.append(")");
        }

        return builder.toString();
    }

    /**
     * his method constructs an equation to describe how a vertex is calculated.
     * The description is generated by recursively stepping up through the BayesNet and generating descriptions.
     * Descriptions of common vertices will use infix operators.
     * Descriptions will not recurse any further than labelled vertices.
     *
     * It is suggested that to use this feature, you label as many vertices as possible to avoid complex descriptions.
     * @param <T> the type of the vertex
     * @param vertex The vertex to create the description of
     * @return An String equation describing how this vertex is calculated.<br>
     * E.g. "This Vertex = that + (three * Const(4))"
     */
    public static <T extends Tensor> String createDescription(Vertex<T> vertex) {
        Collection<Vertex> parents = vertex.getParents();

        if (parents.size() == 0) {
            if (vertex.getLabel() == null) {
                return "<<Unlabelled Leaf Vertex>>";
            }
            return "Vertex with no parents and label: " + vertex.getLabel().toString();
        }

        String thisLabel = vertex.getLabel() != null ? vertex.getLabel().toString() : "This Vertex";

        return thisLabel + " = " + recursiveDescriptionStep(vertex, false);
    }

    private static <T extends Tensor> String getDescriptionDelimiter(Vertex<T> vertex) {
        return delimiters.getOrDefault(vertex.getClass(), ", ");
    }

    private static <T extends Tensor> String getInfixSymbol(Vertex<T> vertex) {
        return infixes.getOrDefault(vertex.getClass(), ", ");
    }
}
