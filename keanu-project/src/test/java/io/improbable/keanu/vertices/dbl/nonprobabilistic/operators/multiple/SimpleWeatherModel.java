package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.apache.commons.io.FileUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class SimpleWeatherModel {

    private final BufferedReader humidityReader;
    private final BufferedReader rainReader;
    private final BufferedReader suggestedFactorSuncreamReader;
    private final BufferedReader isSunnyReader;
    private DoubleVertex inputToModel;

    public SimpleWeatherModel(DoubleVertex inputToModel) throws IOException {
        this.inputToModel = inputToModel;

        humidityReader = mock(BufferedReader.class);
        when(humidityReader .readLine()).thenAnswer(
            invocation -> String.valueOf(blackBoxHumidityModel(this.inputToModel.getValue().scalar()))
        );

        rainReader = mock(BufferedReader.class);
        when(rainReader .readLine()).thenAnswer(
            invocation -> String.valueOf(blackBoxRainModel(this.inputToModel.getValue().scalar()))
        );

        suggestedFactorSuncreamReader = mock(BufferedReader.class);
        when(suggestedFactorSuncreamReader.readLine()).thenAnswer(
            invocation -> String.valueOf(blackBoxSunCreamModel(this.inputToModel.getValue().scalar()))
        );

        isSunnyReader = mock(BufferedReader.class);
        when(isSunnyReader .readLine()).thenAnswer(
            invocation -> String.valueOf(blackBoxIsSunnyModel(this.inputToModel.getValue().scalar()))
        );
    }

    public void modelExecution(Map<VertexLabel, Vertex<? extends Tensor>> inputs) {
        double temperature = inputs.get(new VertexLabel("Temperature")).getValue().asFlatDoubleArray()[0];
        try {
            double chanceOfRain = blackBoxRainModel(temperature);
            double humidity = blackBoxHumidityModel(temperature);
            FileUtils.writeStringToFile(File.createTempFile("chanceOfRainResults", "csv"), String.valueOf(chanceOfRain));
            FileUtils.writeStringToFile(File.createTempFile("humidityResults", "csv"), String.valueOf(humidity));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public Map<VertexLabel, Vertex<? extends Tensor>> updateValues() {
        Map<VertexLabel, Vertex<? extends Tensor>> modelOutput = new HashMap<>();

        try {
            double chanceOfRainResult = Double.parseDouble(getRainReader().readLine());
            modelOutput.put(new VertexLabel("ChanceOfRain"), ConstantVertex.of(chanceOfRainResult));
            double humidityResult = Double.parseDouble(getHumidityReader().readLine());
            modelOutput.put(new VertexLabel("Humidity"), ConstantVertex.of(humidityResult));
        } catch (IOException e) {
            e.printStackTrace();
        }

        return modelOutput;
    }

    public Map<VertexLabel, Vertex<? extends Tensor>> updateValuesMultipleTypes() {
        Map<VertexLabel, Vertex<? extends Tensor>> modelOutput = new HashMap<>();

        try {
            int chanceOfRainResult = (int) Double.parseDouble(getSuggestedFactorSuncreamReader().readLine());
            modelOutput.put(new VertexLabel("suggestedFactorSuncream"), ConstantVertex.of(chanceOfRainResult));
            boolean humidityResult = Boolean.parseBoolean(getIsSunnyReader().readLine());
            modelOutput.put(new VertexLabel("isSunny"), ConstantVertex.of(humidityResult));
        } catch (IOException e) {
            e.printStackTrace();
        }

        return modelOutput;
    }

    public void setInputToModel(DoubleVertex inputValue) {
        this.inputToModel = inputValue;
    }

    private double blackBoxRainModel(double temperature) {
        return temperature * 0.1;
    }

    private double blackBoxHumidityModel(double temperature) {
        return temperature * 2;
    }

    private int blackBoxSunCreamModel(double temperature) {
        return (int) (temperature / 10.0);
    }

    private boolean blackBoxIsSunnyModel(double temperature) {
        return temperature > 20.0;
    }

    public BufferedReader getHumidityReader() {
        return humidityReader;
    }

    public BufferedReader getRainReader() {
        return rainReader;
    }

    public BufferedReader getSuggestedFactorSuncreamReader() {
        return suggestedFactorSuncreamReader;
    }

    public BufferedReader getIsSunnyReader() {
        return isSunnyReader;
    }
}
