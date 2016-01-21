package nettest;

import feedforward.ActivationFunction;
import feedforward.FeedForwardNeuralNetwork;
import org.apache.commons.io.FileUtils;
import org.json.JSONObject;
import training.Backpropagation;
import training.Trainer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class Main
{
        // Array of every JSON file name. Indices are the same as the datasets.
    public static String[] neuralNetFile = {
        "data/chooser1.JSON",//------------------------------> 0
    };
    
    // Array of every file name
    public static String[] dataFile = {//                Index | Description
        "data/600_set1.csv",//---------------------------> 0   | 600 sets run before seeding tournament
    };



    /**
     * [0] => learning rate
     * [1] => momentum
     */
    public static double[] BackpropParams = {
        .01,
        .1
    };

    /**
     * To run a specific training algorithm, specify its parameters in its
     * respective array and select set the fileIndex variable to the index of 
     * the desired dataset (described above).
     * @param args the command line arguments
     *
     *             4.205607476635514
     *             31.682242990654203
     *             37.47663551401869
     */
    public static void main(String[] args)
    {
        int fileIndex = 0;  // Specify the file to use (see file array)
        
        // Initialize Trainer(s)
        Trainer backpropagation = new Backpropagation(BackpropParams);
        
        Trainer[] trainers = new Trainer[] {
            backpropagation,
        };
        
        
        //Nothing below here needs to be changed. 
        //It loads the specified dataset and neural net
        
        
        // Initialize dataset
        double[][] dataset = DataTools.getDataFromFile(dataFile[fileIndex]);
        
        // Load the initial neural net into a JSONObject.
        File initialNet = new File(neuralNetFile[fileIndex]);
        JSONObject json;
        try {
            json = new JSONObject(FileUtils.readFileToString(initialNet));
        } catch (java.io.IOException e1) {
            
            int index = dataset[0].length - 1;
            int numOutputs = 0;
            ArrayList<Double> values = new ArrayList<>();
            
            for (int i = 0; i < dataset.length; i++) {
                Double value = dataset[i][index];
                if (!values.contains(value)) {
                    numOutputs++;
                    values.add(value);
                }
            }
            FeedForwardNeuralNetwork neuralNet = new FeedForwardNeuralNetwork(1, new int[] { dataset[0].length - 1, 10, numOutputs }, ActivationFunction.LOGISTIC, ActivationFunction.LOGISTIC);
            try {
                neuralNet.export(new File(neuralNetFile[fileIndex]));
            } catch(IOException e2) {
                System.err.println(e2.getMessage());
            }
            
            json = neuralNet.export();
        }
        
        // Initialize Experimenter
        Experimenter experiment = new Experimenter(json, trainers);
        
        // Run experiment
        experiment.run(dataset);
        
    }
    
}
