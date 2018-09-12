package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.NonGradientOptimizer;
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

    private KeanuRandom random;


    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void canRunAModelInAModel() {
        DoubleVertex inputToModel = new ConstantDoubleVertex(25);

        Map<VertexLabel, DoubleVertex> inputs = new HashMap<>();
        inputs.put(new VertexLabel("Temperature"), inputToModel);

        String command = "./src/test/resources/model.sh {Temperature}";

        ModelVertex model = new ShellModelVertex(command, inputs, Collections.EMPTY_MAP, this::extractOutput);
        DoubleVertex chanceOfRain = model.getModelOutputVertex(new VertexLabel("ChanceOfRain"));
        DoubleVertex humidity = model.getModelOutputVertex(new VertexLabel("Humidity"));
        DoubleVertex shouldIBringUmbrella = chanceOfRain.times(humidity);

        inputToModel.setAndCascade(10);
        Assert.assertEquals(shouldIBringUmbrella.getValue().scalar(), 20.0, 1e-6);
    }

    @Test
    public void modelInsideVertexIsRecalculatedOnEachParentSample() {
        int numSamples = 50;

        DoubleVertex inputToModel = new ConstantDoubleVertex(25);

        Map<VertexLabel, DoubleVertex> inputs = new HashMap<>();
        inputs.put(new VertexLabel("Temperature"), inputToModel);

        String command = "./src/test/resources/model.sh {Temperature}";

        ModelVertex model = new ShellModelVertex(command, inputs, Collections.EMPTY_MAP, this::extractOutput);
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

    @Test
    public void modelWorksAsPartOfLargerGraph() {
        DoubleVertex inputToModelOne = new GaussianVertex(14.0, 5);
        DoubleVertex inputToModelTwo = new GaussianVertex(14.0, 5);
        DoubleVertex inputToModel = inputToModelOne.plus(inputToModelTwo);

        Map<VertexLabel, DoubleVertex> inputs = new HashMap<>();
        inputs.put(new VertexLabel("Temperature"), inputToModel);

        String command = "./src/test/resources/model.sh {Temperature}";

        ModelVertex model = new ShellModelVertex(command, inputs, Collections.EMPTY_MAP, this::extractOutput);
        DoubleVertex chanceOfRain = model.getModelOutputVertex(new VertexLabel("ChanceOfRain"));
        DoubleVertex humidity = model.getModelOutputVertex(new VertexLabel("Humidity"));

        DoubleVertex temperatureReadingOne = new GaussianVertex(chanceOfRain, 5);
        DoubleVertex temperatureReadingTwo = new GaussianVertex(humidity, 5);
        temperatureReadingOne.observe(3.0);
        temperatureReadingTwo.observe(60.0);

        NonGradientOptimizer gradientOptimizer = NonGradientOptimizer.of(temperatureReadingTwo.getConnectedGraph());
        gradientOptimizer.maxLikelihood();
        Assert.assertEquals(30.0, inputToModel.getValue().scalar(), 0.1);
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
