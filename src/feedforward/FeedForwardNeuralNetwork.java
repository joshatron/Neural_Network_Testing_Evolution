package feedforward;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.*;
import java.util.Random;

/**
 * Created by joshua on 9/22/15.
 */
public class FeedForwardNeuralNetwork
{
    private double momentum;
    private double learningRate;
    private double[] weights;
    private int hiddenLayers;
    private int[] sizes;
    private int biggestSize;
    private ActivationFunction activationFunction;

    public FeedForwardNeuralNetwork(JSONObject net)
    {
        momentum = net.getDouble("momentum");
        learningRate = net.getDouble("learningRate");
        hiddenLayers = net.getInt("hiddenLayers");

        String activation = net.getString("activationFunction");
        if(activation.equals("LINEAR"))
        {
            activationFunction = ActivationFunction.LINEAR;
        }
        else if(activation.equals("LOGISTIC"))
        {
            activationFunction = ActivationFunction.LOGISTIC;
        }
        else
        {
            System.out.println("Illegal activation function.");
        }

        JSONArray sizeArray = net.getJSONArray("sizes");
        if(sizeArray.length() == hiddenLayers + 2)
        {
            sizes = new int[hiddenLayers + 2];
            biggestSize = 0;
            for(int k = 0; k < hiddenLayers + 2; k++)
            {
                int size = sizeArray.getInt(k);
                sizes[k] = size;
                if(size > biggestSize)
                {
                    biggestSize = size;
                }
            }
        }
        else
        {
            System.out.println("Illegal number of layer sizes.");
        }

        if(net.has("weights"))
        {
            int totalWeights = 0;
            for(int k = 0; k < sizes.length - 1; k++)
            {
                totalWeights += sizes[k] * sizes[k + 1];
                totalWeights += sizes[k];
            }
            totalWeights += sizes[sizes.length - 1];
            JSONArray weightArray = net.getJSONArray("weights");
            if(weightArray.length() == totalWeights)
            {
                weights = new double[totalWeights];
                for(int k = 0; k < totalWeights; k++)
                {
                    weights[k] = sizeArray.getInt(k);
                }
            }
            else
            {
                System.out.println("Illegal number of weights");
            }
        }
        else
        {
            generateRandomWeights();
        }
    }

    public FeedForwardNeuralNetwork(int hiddenLayers, int[] sizes, ActivationFunction activationFunction, double momentum, double learningRate)
    {
        this.hiddenLayers = hiddenLayers;
        this.sizes = sizes;
        this.activationFunction = activationFunction;
        this.momentum = momentum;
        this.learningRate = learningRate;

        biggestSize = 0;
        for(int k = 0; k < sizes.length; k++)
        {
            if(sizes[k] > biggestSize)
            {
                biggestSize = sizes[k];
            }
        }

        generateRandomWeights();
    }

    private void generateRandomWeights()
    {
        int totalWeights = 0;
        for(int k = 0; k < sizes.length - 1; k++)
        {
            totalWeights += sizes[k] * sizes[k + 1];
            totalWeights += sizes[k];
        }
        totalWeights += sizes[sizes.length - 1];

        int lowest = -1;
        int highest = 1;
        Random rand = new Random();
        weights = new double[totalWeights];

        for(int k = 0; k < totalWeights; k++)
        {
            weights[k] = lowest + ((highest - lowest) * rand.nextDouble());
        }
    }

    public JSONObject export()
    {
        JSONObject net = new JSONObject();
        net.put("momentum", momentum);
        net.put("learningRate", learningRate);
        net.put("hiddenLayers", hiddenLayers);
        net.put("activationFunction", activationFunction.name());
        net.put("sizes", new JSONArray(sizes));
        net.put("weights", new JSONArray(weights));

        return net;
    }

    public void export(File file) throws IOException
    {
        JSONObject net = export();

        BufferedWriter out = new BufferedWriter(new PrintWriter(new FileWriter(file)));
        out.write(net.toString(4));
    }
}
