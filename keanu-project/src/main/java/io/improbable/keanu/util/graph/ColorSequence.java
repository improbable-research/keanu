package io.improbable.keanu.util.graph;

import java.awt.Color;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class ColorSequence<T> {

    private List<Color> colors = new LinkedList<>();
    private Map<T, Color> assignments = new HashMap<>();

    public ColorSequence(){
        colors.add(Color.RED);
        colors.add(Color.GREEN);
        colors.add(Color.BLUE);
        colors.add(Color.ORANGE);
        colors.add(Color.CYAN);
        colors.add(Color.PINK);
        colors.add(Color.BLACK);
        colors.add(Color.GRAY);
    }

    public Color getOrChoseColor(T src) {
        if ( assignments.containsKey(src) ){
            return assignments.get( src);
        }else{
            Color c = nextColor();
            assignments.put( src , c );
            return c;
        }
    }

    private Color nextColor() {
        return colors.remove(0);
    }
}
