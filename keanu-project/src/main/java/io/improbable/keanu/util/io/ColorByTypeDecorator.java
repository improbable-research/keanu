package io.improbable.keanu.util.io;

import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;

import java.awt.*;
import java.util.Map;

public class ColorByTypeDecorator extends DefaultDecorator {

    private static final Color PROBABILISTIC_COLOR = Color.PINK;
    private static final Color OBSERVED_COLOR = Color.BLUE;
    private static final Color DETERMINISTIC_COLOR = Color.CYAN;

    @Override
    public Map<String, String> getExtraVertexFields(Vertex v) {
        Map<String,String> map = super.getExtraVertexFields(v);
        Color c = getColorFor(v);
        map.put("color",formatColorForDot(c));
        return map;
    }

    private String formatColorForDot(Color c) {
        return "\""+String.format("#%06X", (0xFFFFFF &  c.getRGB()))+"\"";
    }


    private Color getColorFor(Vertex v) {
        if ( v.isObserved() ) {
            return OBSERVED_COLOR;
        }else if ( v instanceof Probabilistic ){
            return PROBABILISTIC_COLOR;
        }else{
            return DETERMINISTIC_COLOR;
        }
    }
}
