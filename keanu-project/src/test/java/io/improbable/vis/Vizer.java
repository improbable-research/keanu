package io.improbable.vis;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.statistics.HistogramDataset;
import org.jfree.data.statistics.HistogramType;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

import java.awt.*;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Vizer {

    enum PlotType {
        SCATTER, AREA_CHART
    }

    private static JFreeChart createHistogram(List<Double> samples) {

        double[] data = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            data[i] = samples.get(i);
        }

        // int number = data.length;
        HistogramDataset dataset = new HistogramDataset();
        dataset.setType(HistogramType.RELATIVE_FREQUENCY);
        dataset.addSeries("Hist", data, 200);
        String plotTitle = "";
        String xAxis = "Value";
        String yAxis = "Frequency";
        PlotOrientation orientation = PlotOrientation.VERTICAL;

        boolean show = false;
        boolean toolTips = false;
        boolean urls = false;

        JFreeChart chart = ChartFactory.createHistogram(plotTitle, xAxis, yAxis,
                dataset, orientation, show, toolTips, urls);

        chart.setBackgroundPaint(Color.white);

        return chart;
    }

    private static JFreeChart createPlot(List<Double> x, List<Double> y, PlotType plotType) {
        if (x.size() != y.size()) {
            throw new IllegalArgumentException();
        }

        XYSeries dataset = new XYSeries("");

        for (int i = 0; i < x.size(); i++) {
            dataset.add(x.get(i), y.get(i));
        }

        String plotTitle = "";
        String xAxis = "x";
        String yAxis = "y";
        PlotOrientation orientation = PlotOrientation.VERTICAL;

        XYSeriesCollection collection = new XYSeriesCollection();
        collection.addSeries(dataset);

        boolean show = false;
        boolean toolTips = false;
        boolean urls = false;
        JFreeChart chart;

        switch (plotType) {
            case SCATTER:
                chart = ChartFactory.createScatterPlot(plotTitle, xAxis, yAxis,
                        collection, orientation, show, toolTips, urls);
                break;
            default:
                chart = ChartFactory.createXYAreaChart(plotTitle, xAxis, yAxis,
                        collection, orientation, show, toolTips, urls);
                break;

        }

        chart.setBackgroundPaint(Color.white);

        return chart;
    }

    private static JFreeChart createPlot(List<Double> y) {

        List<Double> x = IntStream.range(0, y.size())
                .asDoubleStream().boxed()
                .collect(Collectors.toList());

        return createPlot(x, y, PlotType.AREA_CHART);
    }

    public static void histogram(List<Double> samples) {
        histogram(samples, "Vizer");
    }

    public static void histogram(List<Double> samples, String title) {
        display(createHistogram(samples), title);
    }

    public static void scatter(List<Double> x, List<Double> y, String title) {
        display(createPlot(x, y, PlotType.SCATTER), title);
    }

    public static void plot(List<Double> y, String title) {
        display(createPlot(y), title);
    }

    private static void display(JFreeChart chart, String title) {
        ApplicationFrame frame = new ApplicationFrame(title);
        final ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));
        frame.setContentPane(chartPanel);

        frame.pack();
        RefineryUtilities.centerFrameOnScreen(frame);
        frame.setVisible(true);
    }

}
