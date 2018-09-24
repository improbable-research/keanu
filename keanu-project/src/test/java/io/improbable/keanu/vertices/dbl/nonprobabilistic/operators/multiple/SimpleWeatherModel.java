package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.vertices.dbl.DoubleVertex;

import java.io.BufferedReader;
import java.io.IOException;

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

    public void setInputToModel(DoubleVertex inputValue) {
        this.inputToModel = inputValue;
    }

    public double blackBoxRainModel(double temperature) {
        return temperature * 0.1;
    }

    public double blackBoxHumidityModel(double temperature) {
        return temperature * 2;
    }

    public int blackBoxSunCreamModel(double temperature) {
        return (int) (temperature / 10.0);
    }

    public boolean blackBoxIsSunnyModel(double temperature) {
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
