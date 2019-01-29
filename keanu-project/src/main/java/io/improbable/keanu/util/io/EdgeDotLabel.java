package io.improbable.keanu.util.io;

import org.apache.commons.lang3.StringUtils;

import java.util.Collection;
import java.util.Map;

public class EdgeDotLabel {

    private static final String DOT_LABEL_OPENING = " [label=";
    private static final String DOT_LABEL_CLOSING = "]";
    private static final String DOT_FIELD_OPENING = " [";
    private static final String DOT_FIELD_SEPARATOR = "=";
    private static final String DOT_FIELD_CLOSING = "]";

    public static String inDotFormat(GraphEdge edge, DotDecorator decorator) {
        String dotOutput = edge.getParentVertex().hashCode() + " -> " + edge.getChildVertex().hashCode();
        Collection<String> labels = decorator.labelEdge(edge);
        if (!labels.isEmpty()) {
            dotOutput += DOT_LABEL_OPENING;
            dotOutput += StringUtils.join(labels, ", ");
            dotOutput += DOT_LABEL_CLOSING;
        }
        Map<String,String> fields = decorator.getExtraEdgeFields(edge);
        for ( Map.Entry<String,String> e : fields.entrySet() ){
            dotOutput += DOT_FIELD_OPENING;
            dotOutput += e.getKey();
            dotOutput += DOT_FIELD_SEPARATOR;
            dotOutput += e.getValue();
            dotOutput += DOT_FIELD_CLOSING;
        }
        return dotOutput;
    }

}
