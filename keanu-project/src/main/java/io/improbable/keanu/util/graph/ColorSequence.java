package io.improbable.keanu.util.graph;

import java.awt.*;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * This class provides a maping from elements of T into unique colors.
 * It's main use is applying coloring to graphs.
 * @param <T> Input type
 */
public class ColorSequence<T> {

    private List<Color> colors = new LinkedList<>();
    private Map<T, Color> assignments = new HashMap<>();

    public ColorSequence() {
        colors.add(Color.RED);
        colors.add(Color.GREEN);
        colors.add(Color.BLUE);
        colors.add(Color.ORANGE);
        colors.add(Color.CYAN);
        colors.add(Color.PINK);
        colors.add(Color.BLACK);
        colors.add(Color.GRAY);
        List<Color> extraColors = colors.stream().map((c) -> generateAlternateColours(c, 0.5f)).collect(Collectors.toList());
        extraColors.addAll( colors.stream().map((c) -> generateAlternateColours(c, 1.5f)).collect(Collectors.toList()) );
        colors.addAll(extraColors);
        extraColors = colors.stream().map((c) -> generateAlternateColours(c, 0.8f)).collect(Collectors.toList());
        extraColors.addAll( colors.stream().map((c) -> generateAlternateColours(c, 1.2f)).collect(Collectors.toList()) );
        colors.addAll(extraColors);
    }

    private static final float[] comps = new float[4];

    private static Color generateAlternateColours(Color c, float delta) {
        c.getComponents( comps );
        for ( int i =0;i<4;i++){
            comps[i] = Math.min(Math.max(comps[i] * delta,0.0f),1.0f);
        }
        return new Color( comps[0] , comps[1] , comps[2]  );
    }

    /**
     * Primary usage
     * @param src Source value to lookup
     * @return a Color that represents that value
     */
    public Color getOrChoseColor(T src) {
        if (src == null) return null;
        if (assignments.containsKey(src)) {
            return assignments.get(src);
        } else {
            Color c = nextColor();
            assignments.put(src, c);
            System.err.println("Assigning "+c+" to "+src);
            return c;
        }
    }

    private Color nextColor() {
        return colors.remove(0);
    }
}
