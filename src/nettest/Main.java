package nettest;

import java.io.File;
import training.*;
import org.json.JSONObject;
import org.apache.commons.io.FileUtils;

public class Main
{
    /**
     * [0] => # of children
     * [1] => # of generations
     * [2] => mutation rate
     * [3] => cross over rate
     * [4] => populationSize
     * [5] => # of weights
     * [6] => crossover Type
     */
    public static double[] GAParams = {
            50,
            100,
            0.01,
            0.1,
            50,
            5,
            0
    };

    /**
     * 
     * @param args the command line arguments
     */
    public static void main(String[] args)
    {
        
        // Load all of the datasets and put them into one 3D double array
        double[][] dataset1 = new double[][]{{1.0}};
        double[][] dataset2 = new double[][]{{1.0}};
        double[][] dataset3 = new double[][]{{1.0}};
        double[][][] datasets = new double[][][]{dataset1,dataset2,dataset3};
        
        // Load the initial neural net into a JSONObject.
        File initialNet = new File("neuralNet.JSON");
        JSONObject json = null;
        try {
            json = new JSONObject(FileUtils.readFileToString(initialNet));
        } catch (java.io.IOException e) {
            System.err.println(e.getMessage());
        }
        
        // Initialize Trainer(s)
        Trainer geneticAlgorithim = new GeneticAlgorithm(GAParams);
        
        // Initialize Experimenter
        Experimenter experiment = new Experimenter(json, new Trainer[]{new GeneticAlgorithm(GAParams)});
        
        // Run experiment
        experiment.run(datasets);
        
    }
    
}
