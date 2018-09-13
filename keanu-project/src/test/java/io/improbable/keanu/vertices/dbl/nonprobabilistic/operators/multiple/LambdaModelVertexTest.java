package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.NonGradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.invocation.InvocationOnMock;
import org.mockito.stubbing.Answer;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class LambdaModelVertexTest {

    @Mock
    private BufferedReader rainReader;

    @Mock
    private BufferedReader humidityReader;

    private KeanuRandom random;
    private DoubleVertex inputToModel;

    @Before
    public void setup() throws IOException {
        random = new KeanuRandom(1);
        rainReader = mock(BufferedReader.class);
        humidityReader = mock(BufferedReader.class);

        when(rainReader.readLine()).thenAnswer(new Answer<Object>() {
            @Override
            public Object answer(InvocationOnMock invocation) {
                double chanceOfRainScalingFactorFromModel = 0.1;
                return String.valueOf(inputToModel.getValue().scalar() * chanceOfRainScalingFactorFromModel);
            }
        });

        when(humidityReader.readLine()).thenAnswer(new Answer<Object>() {
            @Override
            public Object answer(InvocationOnMock invocation) {
                double humidityScalingFactorFromModel = 2;
                return String.valueOf(inputToModel.getValue().scalar() * humidityScalingFactorFromModel);
            }
        });
    }

    @Test
    public void canRunAModelInAModel() {
        inputToModel = new ConstantDoubleVertex(25);
        Map<VertexLabel, DoubleVertex> inputs = new HashMap<>();
        inputs.put(new VertexLabel("Temperature"), inputToModel);


        ModelVertex model = new LambdaModelVertex(inputs, this::modelExecution, this::updateValues);
        DoubleVertex chanceOfRain = model.getModelOutputVertex(new VertexLabel("ChanceOfRain"));
        DoubleVertex humidity = model.getModelOutputVertex(new VertexLabel("Humidity"));

        DoubleVertex shouldIBringUmbrella = chanceOfRain.times(humidity);

        double inputValue = 10.0;

        inputToModel.setAndCascade(inputValue);
        Assert.assertEquals(shouldIBringUmbrella.getValue().scalar(), 20.0, 1e-6);
    }

    @Test
    public void modelInsideVertexIsRecalculatedOnEachParentSample() {
        int numSamples = 50;

        inputToModel = new ConstantDoubleVertex(25);
        Map<VertexLabel, DoubleVertex> inputs = new HashMap<>();
        inputs.put(new VertexLabel("Temperature"), inputToModel);

        ModelVertex model = new LambdaModelVertex(inputs, this::modelExecution, this::updateValues);
        DoubleVertex chanceOfRain = model.getModelOutputVertex(new VertexLabel("ChanceOfRain"));
        DoubleVertex humidity = model.getModelOutputVertex(new VertexLabel("Humidity"));
        DoubleVertex shouldIBringUmbrella = chanceOfRain.times(humidity);

        for (int i = 0; i < numSamples; i++) {
            double inputValue = inputToModel.sample(random).scalar();
            inputToModel.setAndCascade(inputValue);
            double expectedValue = (inputValue * 0.1) * (inputValue * 2);
            Assert.assertEquals(expectedValue, shouldIBringUmbrella.getValue().scalar(), 1e-6);
        }
    }

    @Test
    public void modelWorksAsPartOfGradientOptimisation() {
        DoubleVertex inputToModelOne = new GaussianVertex(14.0, 5);
        DoubleVertex inputToModelTwo = new GaussianVertex(14.0, 5);
        inputToModel = inputToModelOne.plus(inputToModelTwo);

        Map<VertexLabel, DoubleVertex> inputs = new HashMap<>();
        inputs.put(new VertexLabel("Temperature"), inputToModel);

        ModelVertex model = new LambdaModelVertex(inputs, this::modelExecution, this::updateValues);
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

    @Test
    public void modelWorksAsPartOfSampling() {
        inputToModel = new GaussianVertex(29., 2);

        Map<VertexLabel, DoubleVertex> inputs = new HashMap<>();
        inputs.put(new VertexLabel("Temperature"), inputToModel);

        ModelVertex model = new LambdaModelVertex(inputs, this::modelExecution, this::updateValues);
        DoubleVertex chanceOfRain = model.getModelOutputVertex(new VertexLabel("ChanceOfRain"));
        DoubleVertex humidity = model.getModelOutputVertex(new VertexLabel("Humidity"));

        DoubleVertex temperatureReadingOne = new GaussianVertex(chanceOfRain, 2);
        DoubleVertex temperatureReadingTwo = new GaussianVertex(humidity, 2);
        //My prior belief is the temperature is 29.0. These observations are indicative of 30.
        temperatureReadingOne.observe(3.0);
        temperatureReadingTwo.observe(60.0);

        BayesianNetwork bayesianNetwork = new BayesianNetwork(temperatureReadingOne.getConnectedGraph());

        NetworkSamples posteriorSamples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
            bayesianNetwork,
            inputToModel,
            50000
        );

        double averagePosteriorInput = posteriorSamples.getDoubleTensorSamples(inputToModel).getAverages().scalar();

        Assert.assertEquals((29 * (1 / 3.) + (30 * (2 / 3.))), averagePosteriorInput, 0.1);
    }

    private void modelExecution(Map<VertexLabel, DoubleVertex> inputs) {
        //This is a placeholder for the function that would write to file.
        //Instead we are mocking the file IO
    }

    private Map<VertexLabel, Double> updateValues(Map<VertexLabel, DoubleVertex> inputs) {
        Map<VertexLabel, Double> modelOutput = new HashMap<>();

        try {
            double chanceOfRainResult = Double.parseDouble(rainReader.readLine());
            modelOutput.put(new VertexLabel("ChanceOfRain"), chanceOfRainResult);
            double humidityResult = Double.parseDouble(humidityReader.readLine());
            modelOutput.put(new VertexLabel("Humidity"), humidityResult);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return modelOutput;
    }

}
