package io.improbable.keanu.util.dot;

import org.apache.commons.lang3.StringUtils;

public class EdgeDotLabel {

    private static final String DOT_LABEL_OPENING = " [label=";
    private static final String DOT_LABEL_CLOSING = "]";

    public static String inDotFormat(GraphEdge edge) {
        String dotOutput = "<" + edge.getParentVertex().hashCode() + "> -> <" + edge.getChildVertex().hashCode() + ">";
        if (!edge.getLabels().isEmpty()) {
            dotOutput += DOT_LABEL_OPENING;
            dotOutput += StringUtils.join(edge.getLabels(), ", ");
            dotOutput += DOT_LABEL_CLOSING;
        }

        return dotOutput;
    }

}
