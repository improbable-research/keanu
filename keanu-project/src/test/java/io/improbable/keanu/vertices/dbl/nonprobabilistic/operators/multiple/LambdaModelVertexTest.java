package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.NonGradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.model.LambdaModelVertex;
import io.improbable.keanu.vertices.model.ModelVertex;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class LambdaModelVertexTest {

    /*
    The model we are mimicking here lives inside a Java function, modelExecution.

    It takes one input, Temperature, and produces two outputs of type double, Chance of Rain & Humidity. These outputs
    are written to file.

    It also produces one integer output and one boolean output. Suggested Factor of Suncream and 'is it sunny'. These
    are also written to file.
     */

    private KeanuRandom random;
    private DoubleVertex inputToModel;
    private SimpleWeatherModel weatherModel;

    @Before
    public void mockFilesToReadModelOutput() throws IOException {
        random = new KeanuRandom(1);
        weatherModel = new SimpleWeatherModel(inputToModel);
        inputToModel = new ConstantDoubleVertex(25.);
    }

    @Test
    public void canRunAModelInAModel() {
        weatherModel.setInputToModel(inputToModel);
        Map<VertexLabel, Vertex<? extends Tensor>> inputs = ImmutableMap.of(new VertexLabel("Temperature"), inputToModel);

        ModelVertex model = new LambdaModelVertex(inputs, weatherModel::modelExecution, weatherModel::updateValues);
        DoubleVertex chanceOfRain = model.getDoubleModelOutputVertex(new VertexLabel("ChanceOfRain"));
        DoubleVertex humidity = model.getDoubleModelOutputVertex(new VertexLabel("Humidity"));

        DoubleVertex shouldIBringUmbrella = chanceOfRain.times(humidity);

        double inputValue = 10.0;

        inputToModel.setAndCascade(inputValue);
        Assert.assertEquals(shouldIBringUmbrella.getValue().scalar(), 20.0, 1e-6);
    }

    @Test
    public void canRunEvalOnTheOutputsToRecalculateTheModel() {
        weatherModel.setInputToModel(inputToModel);
        Map<VertexLabel, Vertex<? extends Tensor>> inputs = ImmutableMap.of(new VertexLabel("Temperature"), inputToModel);

        ModelVertex model = new LambdaModelVertex(inputs, weatherModel::modelExecution, weatherModel::updateValues);
        DoubleVertex chanceOfRain = model.getDoubleModelOutputVertex(new VertexLabel("ChanceOfRain"));
        DoubleVertex humidity = model.getDoubleModelOutputVertex(new VertexLabel("Humidity"));

        DoubleVertex shouldIBringUmbrella = chanceOfRain.times(humidity);

        inputToModel.setValue(10.0);
        shouldIBringUmbrella.eval();
        Assert.assertEquals(20.0, shouldIBringUmbrella.getValue().scalar(), 1e-6);

        inputToModel.setValue(20.0);
        shouldIBringUmbrella.eval();
        Assert.assertEquals(80.0, shouldIBringUmbrella.getValue().scalar(), 1e-6);
    }

    @Test
    public void canRunAModelInAModelWithDifferentOutputTypes() {
        weatherModel.setInputToModel(inputToModel);
        Map<VertexLabel, Vertex<? extends Tensor>> inputs = ImmutableMap.of(new VertexLabel("Temperature"), inputToModel);

        ModelVertex model = new LambdaModelVertex(inputs, weatherModel::modelExecution, weatherModel::updateValuesMultipleTypes);
        IntegerVertex suggestedFactorSuncream = model.getIntegerModelOutputVertex(new VertexLabel("suggestedFactorSuncream"));
        BoolVertex isSunny = model.getBoolModelOutputVertex(new VertexLabel("isSunny"));

        double inputValue = 20.0;

        inputToModel.setAndCascade(inputValue);
        Assert.assertEquals(suggestedFactorSuncream.getValue().scalar(), new Integer(2));
        Assert.assertEquals(isSunny.getValue().scalar(), false);
    }

    @Test
    public void modelInsideVertexIsRecalculatedOnEachParentSample() {
        int numSamples = 50;
        weatherModel.setInputToModel(inputToModel);
        Map<VertexLabel, Vertex<? extends Tensor>> inputs = ImmutableMap.of(new VertexLabel("Temperature"), inputToModel);

        ModelVertex model = new LambdaModelVertex(inputs, weatherModel::modelExecution, weatherModel::updateValues);
        DoubleVertex chanceOfRain = model.getDoubleModelOutputVertex(new VertexLabel("ChanceOfRain"));
        DoubleVertex humidity = model.getDoubleModelOutputVertex(new VertexLabel("Humidity"));
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
        weatherModel.setInputToModel(inputToModel);

        Map<VertexLabel, Vertex<? extends Tensor>> inputs = ImmutableMap.of(new VertexLabel("Temperature"), inputToModel);

        ModelVertex model = new LambdaModelVertex(inputs, weatherModel::modelExecution, weatherModel::updateValues);
        DoubleVertex chanceOfRain = model.getDoubleModelOutputVertex(new VertexLabel("ChanceOfRain"));
        DoubleVertex humidity = model.getDoubleModelOutputVertex(new VertexLabel("Humidity"));

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
        inputToModel = new GaussianVertex(25., 5);
        weatherModel.setInputToModel(inputToModel);

        Map<VertexLabel, Vertex<? extends Tensor>> inputs = ImmutableMap.of(new VertexLabel("Temperature"), inputToModel);

        ModelVertex model = new LambdaModelVertex(inputs, weatherModel::modelExecution, weatherModel::updateValues);
        DoubleVertex chanceOfRain = model.getDoubleModelOutputVertex(new VertexLabel("ChanceOfRain"));
        DoubleVertex humidity = model.getDoubleModelOutputVertex(new VertexLabel("Humidity"));

        //My prior belief is the temperature is 29.0.
        //These observations are indicative of a temperature of 30.
        DoubleVertex chanceOfRainObservation = new GaussianVertex(chanceOfRain, 5);
        DoubleVertex humidityObservation = new GaussianVertex(humidity, 5);
        humidityObservation.observe(60.0);
        chanceOfRainObservation.observe(3.0);

        BayesianNetwork bayesianNetwork = new BayesianNetwork(chanceOfRainObservation.getConnectedGraph());

        NetworkSamples posteriorSamples = MetropolisHastings.withDefaultConfig(random).getPosteriorSamples(
            bayesianNetwork,
            inputToModel,
            2500
        );

        double averagePosteriorInput = posteriorSamples.getDoubleTensorSamples(inputToModel).getAverages().scalar();

        Assert.assertEquals(29., averagePosteriorInput, 0.1);
    }

}
