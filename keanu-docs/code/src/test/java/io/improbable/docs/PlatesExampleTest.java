package io.improbable.docs;


import io.improbable.keanu.plating.Plate;
import io.improbable.keanu.plating.Plates;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import org.junit.Test;

import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertThat;

public class PlatesExampleTest {

    @Test
    public void youCanAccessAVertexByPlateNumberAndVertexName() {
        Plates plates = new PlatesExample().buildPlates("plates_example_data.csv");
        Plate plate1 = plates.asList().get(1);
        Vertex<DoubleTensor> x = plate1.get(new VertexLabel("x"));
        assertThat(x.getValue().scalar(), equalTo(0.2));
    }
}
