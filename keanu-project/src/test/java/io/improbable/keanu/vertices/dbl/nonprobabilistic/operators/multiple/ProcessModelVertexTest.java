package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.Keanu;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.NonGradientOptimizer;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.model.LambdaModelVertex;
import io.improbable.keanu.vertices.model.ModelVertex;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.io.IOException;
import java.util.Map;
import java.util.regex.Pattern;

import static io.improbable.keanu.Keanu.Sampling.MetropolisHastings;

public class ProcessModelVertexTest {

    /*
      The model we are mimicking here is a python script, model.py.

      It takes one input, Temperature, and produces two outputs of type double, Chance of Rain & Humidity. These outputs
      are written to file.

      It also produces one integer output and one boolean output. Suggested Factor of Suncream and 'is it sunny'. These
      are also written to file.

      This test provides an example of how to make use of a model implemented in an external application.  At the moment
      though we just simulate this and the model is actually run in Java code, with our external application just being
      chosen for speed of execution.
     */

    private DoubleVertex inputToModel;
    private SimpleWeatherModel weatherModel;
    private static final String COMMAND = "echo";

    @Rule
    public DeterministicRule rule = new DeterministicRule();

    @Before
    public void mockFilesToReadModelOutputFrom() throws IOException {
        weatherModel = new SimpleWeatherModel(inputToModel);
        inputToModel = new ConstantDoubleVertex(25.0);
    }

    @Test
    public void canRunAModelInAModel() {
        weatherModel.setInputToModel(inputToModel);
        Map<VertexLabel, Vertex<? extends Tensor>> inputs = ImmutableMap.of(new VertexLabel("Temperature"), inputToModel);

        // An example of what running a real Python model would look like
        String command = formatCommandForExecution(inputs, "python ./src/test/resources/model.py {Temperature}");
        ModelVertex model = LambdaModelVertex.createFromProcess(inputs, command, weatherModel::updateValues);

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

        ModelVertex model = LambdaModelVertex.createFromProcess(inputs, COMMAND, weatherModel::updateValues);

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

        ModelVertex model = LambdaModelVertex.createFromProcess(inputs, COMMAND, weatherModel::updateValuesMultipleTypes);

        IntegerVertex suggestedFactorSuncream = model.getIntegerModelOutputVertex(new VertexLabel("suggestedFactorSuncream"));
        BooleanVertex isSunny = model.getBooleanModelOutputVertex(new VertexLabel("isSunny"));

        double inputValue = 20.0;

        inputToModel.setAndCascade(inputValue);
        Assert.assertEquals(suggestedFactorSuncream.getValue().scalar(), new Integer(2));
        Assert.assertEquals(isSunny.getValue().scalar(), false);
    }

    @Category(Slow.class)
    @Test
    public void modelInsideVertexIsRecalculatedOnEachParentSample() {
        int numSamples = 50;
        GaussianVertex probabilisticInput = new GaussianVertex(21., 1.);
        weatherModel.setInputToModel(probabilisticInput);
        Map<VertexLabel, Vertex<? extends Tensor>> inputs = ImmutableMap.of(new VertexLabel("Temperature"), probabilisticInput);

        ModelVertex model = LambdaModelVertex.createFromProcess(inputs, COMMAND, weatherModel::updateValues);

        DoubleVertex chanceOfRain = model.getDoubleModelOutputVertex(new VertexLabel("ChanceOfRain"));
        DoubleVertex humidity = model.getDoubleModelOutputVertex(new VertexLabel("Humidity"));
        DoubleVertex shouldIBringUmbrella = chanceOfRain.times(humidity);

        for (int i = 0; i < numSamples; i++) {
            double inputValue = probabilisticInput.sample().scalar();
            probabilisticInput.setAndCascade(inputValue);
            double expectedValue = (inputValue * 0.1) * (inputValue * 2);
            Assert.assertEquals(expectedValue, shouldIBringUmbrella.getValue().scalar(), 1e-6);
        }
    }

    @Category(Slow.class)
    @Test
    public void modelWorksAsPartOfGradientOptimisation() {
        DoubleVertex inputToModelOne = new GaussianVertex(14.0, 5);
        DoubleVertex inputToModelTwo = new GaussianVertex(14.0, 5);
        inputToModel = inputToModelOne.plus(inputToModelTwo);
        weatherModel.setInputToModel(inputToModel);

        Map<VertexLabel, Vertex<? extends Tensor>> inputs = ImmutableMap.of(new VertexLabel("Temperature"), inputToModel);

        ModelVertex model = LambdaModelVertex.createFromProcess(inputs, COMMAND, weatherModel::updateValues);
        DoubleVertex chanceOfRain = model.getDoubleModelOutputVertex(new VertexLabel("ChanceOfRain"));
        DoubleVertex humidity = model.getDoubleModelOutputVertex(new VertexLabel("Humidity"));

        DoubleVertex temperatureReadingOne = new GaussianVertex(chanceOfRain, 5);
        DoubleVertex temperatureReadingTwo = new GaussianVertex(humidity, 5);
        temperatureReadingOne.observe(3.0);
        temperatureReadingTwo.observe(60.0);

        NonGradientOptimizer nonGradientOptimizer = Keanu.Optimizer.NonGradient.of(temperatureReadingTwo.getConnectedGraph());
        nonGradientOptimizer.maxLikelihood();
        Assert.assertEquals(30.0, inputToModel.getValue().scalar(), 0.1);
    }

    @Category(Slow.class)
    @Test
    public void modelWorksAsPartOfSampling() {
        inputToModel = new GaussianVertex(25, 5);
        weatherModel.setInputToModel(inputToModel);
        Map<VertexLabel, Vertex<? extends Tensor>> inputs = ImmutableMap.of(new VertexLabel("Temperature"), inputToModel);

        ModelVertex model = LambdaModelVertex.createFromProcess(inputs, COMMAND, weatherModel::updateValues);

        DoubleVertex chanceOfRain = model.getDoubleModelOutputVertex(new VertexLabel("ChanceOfRain"));
        DoubleVertex humidity = model.getDoubleModelOutputVertex(new VertexLabel("Humidity"));

        DoubleVertex chanceOfRainObservation = new GaussianVertex(chanceOfRain, 5);
        DoubleVertex humidityObservation = new GaussianVertex(humidity, 5);
        chanceOfRainObservation.observe(3.0);
        humidityObservation.observe(60.0);

        KeanuProbabilisticModel probabilisticModel = new KeanuProbabilisticModel(chanceOfRainObservation.getConnectedGraph());

        NetworkSamples posteriorSamples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
            probabilisticModel,
            inputToModel,
            220
        );

        double averagePosteriorInput = posteriorSamples.getDoubleTensorSamples(inputToModel).getAverages().scalar();

        Assert.assertEquals(29., averagePosteriorInput, 0.3);
    }

    private String formatCommandForExecution(Map<VertexLabel, Vertex<? extends Tensor>> inputs, String command) {
        for (Map.Entry<VertexLabel, io.improbable.keanu.vertices.Vertex<? extends Tensor>> input : inputs.entrySet()) {
            String argument = "{" + input.getKey().toString() + "}";
            command = command.replaceAll(Pattern.quote(argument), input.getValue().getValue().scalar().toString());
        }
        return command;
    }

}
