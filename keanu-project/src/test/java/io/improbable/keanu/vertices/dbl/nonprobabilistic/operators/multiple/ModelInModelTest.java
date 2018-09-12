package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.hamcrest.MatcherAssert;
import org.hamcrest.Matchers;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.*;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

public class ModelInModelTest {

    private DoubleVertex inputToModel;
    private ModelVertex model;
    private KeanuRandom random;


    @Before
    public void setup() {
        random = new KeanuRandom(1);
        inputToModel = new ConstantDoubleVertex(25);

        Map<VertexLabel, DoubleVertex> inputs = new HashMap<>();
        inputs.put(new VertexLabel("Temperature"), inputToModel);

        String command = "./src/test/resources/model.sh {Temperature}";

        model = new ShellModelVertex(command, inputs, Collections.EMPTY_MAP, this::extractOutput);
    }

    @Test
    public void canRunAModelInAModel() {
        DoubleVertex chanceOfRain = model.getModelOutputVertex(new VertexLabel("ChanceOfRain"));
        DoubleVertex humidity = model.getModelOutputVertex(new VertexLabel("Humidity"));
        DoubleVertex shouldIBringUmbrella = chanceOfRain.times(humidity);

        inputToModel.setAndCascade(10);
        Assert.assertEquals(shouldIBringUmbrella.getValue().scalar(), 20.0, 1e-6);
    }

    @Test
    public void modelInsideVertexIsRecalculatedOnEachParentSample() {
        int numSamples = 50;

        DoubleVertex chanceOfRain = model.getModelOutputVertex(new VertexLabel("ChanceOfRain"));
        DoubleVertex humidity = model.getModelOutputVertex(new VertexLabel("Humidity"));
        DoubleVertex shouldIBringUmbrella = chanceOfRain.times(humidity);

        for (int i = 0; i < numSamples; i++) {
            double value = inputToModel.sample(random).scalar();
            inputToModel.setAndCascade(value);
            //"model" logic
            double expectedValue = (value * 0.1) * (value * 2);
            Assert.assertEquals(expectedValue, shouldIBringUmbrella.getValue().scalar(), 1e-6);
        }
    }

    private Map<VertexLabel, Double> extractOutput(Map<VertexLabel, DoubleVertex> inputs) {
        Map<VertexLabel, Double> modelOutput = new HashMap<>();

        try (BufferedReader br = new BufferedReader(new FileReader("src/test/resources/rainOutput.txt"))) {
            double chanceOfRainResult = Double.parseDouble(br.readLine());
            modelOutput.put(new VertexLabel("ChanceOfRain"), chanceOfRainResult);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        try (BufferedReader br = new BufferedReader(new FileReader("src/test/resources/humidityOutput.txt"))) {
            double humidityResult = Double.parseDouble(br.readLine());
            modelOutput.put(new VertexLabel("Humidity"), humidityResult);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return modelOutput;
    }

}
