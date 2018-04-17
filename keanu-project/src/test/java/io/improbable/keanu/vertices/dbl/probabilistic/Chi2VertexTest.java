package io.improbable.keanu.vertices.dbl.probabilistic;

import org.junit.Assert;
import org.junit.Test;

public class Chi2VertexTest {

    @Test
    public void canCreateChiVertex() {
        Chi2Vertex chi = new Chi2Vertex(2);
        double x = chi.sample();
        Assert.assertTrue(x > 0);
    }

    @Test
    public void canGetDensityOfChi() {
        Chi2Vertex chi = new Chi2Vertex(3);
        double a = chi.density(0.0);
        double b = chi.density(1.0);
        Assert.assertTrue(a < b);
    }

    @Test
    public void canGetLogDensityOfChi() {
        Chi2Vertex chi = new Chi2Vertex(2);
        double a = chi.logDensity(0.1);
        System.out.println(a);
    }

    @Test
    public void testLogDensityEqualsLogOfDensity() {
        Chi2Vertex chi = new Chi2Vertex(1);
        chi.setValue(0.0);
        double density = chi.density(0.1);
        double logDensity = chi.logDensity(0.1);

        Assert.assertEquals(Math.log(density), logDensity, 0.001);
    }

}
