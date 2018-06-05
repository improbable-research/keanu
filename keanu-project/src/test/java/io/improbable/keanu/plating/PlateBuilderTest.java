package io.improbable.keanu.plating;

import io.improbable.keanu.vertices.booltensor.probabilistic.Flip;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class PlateBuilderTest {

    private static class Bean {
        public int x;

        public Bean(int x) {
            this.x = x;
        }
    }

    private static final List<Bean> ROWS = Arrays.asList(
        new Bean(0),
        new Bean(0),
        new Bean(0)
    );

    @Test
    public void buildPlatesFromCount_Size() {
        int n = 100;
        Plates plates = new PlateBuilder()
            .count(n)
            .withFactory(plate -> {
            })
            .build();
        assertEquals(n, plates.size());
    }

    @Test
    public void buildPlatesFromCount_PlateContents() {
        int n = 100;
        String vertexName = "vertexName";
        Plates plates = new PlateBuilder<>()
            .count(n)
            .withFactory((plate) -> plate.add(vertexName, new Flip(0.5)))
            .build();
        plates.asList().forEach(plate -> {
            assertNotNull(plate.get(vertexName));
        });
    }

    @Test
    public void buildPlatesFromData_Size() {
        Plates plates = new PlateBuilder<Bean>()
            .fromIterator(ROWS.iterator())
            .withFactory((plate, bean) -> {
            })
            .build();
        assertEquals(ROWS.size(), plates.size());
    }

    @Test
    public void buildPlatesFromData_Contents() {
        Plates plates = new PlateBuilder<Bean>()
            .fromIterator(ROWS.iterator())
            .withFactory((plate, bean) -> {
                assertEquals(0, bean.x);
            })
            .build();
    }


}
